"""Write Alaala's matches back into Apple Photos as keywords, so the normal
Photos search bar (and your iPhone, via iCloud Photos) can find "halo-halo".

    python tag.py --calibrate              # optional: pick the cutoff from your eval labels
    python tag.py                          # propose tags and open the review page
    python tag.py --apply                  # write the tags you approved into Photos
    python tag.py --undo                   # remove every keyword Alaala added

Nothing is written until you review it. Every tagged photo also gets the keyword
"Alaala", so you can find them all in Photos (Smart Album > Keyword is Alaala).

For --apply/--undo, Photos must be open on the same library you indexed.
macOS will ask once to let Terminal control Photos: click OK.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from common import DATA
from culture import CONFIG, norm
from search import KEEP_ORIGINAL, Searcher, thumb_data_uri

THRESHOLD_FILE = DATA / "tag_threshold.json"
APPLIED = DATA / "applied_tags.json"
DEFAULT_Z = 3.0      # how far above a photo's average score it must be (in std devs)
MAX_PER_TERM = 40
MARKER = "Alaala"


def zscores(s: Searcher, term: str, learned: bool = True) -> np.ndarray:
    """Score every photo for a term, normalized so one cutoff works for all terms.
    Uses what Alaala has learned from your feedback, when it has learned something."""
    raw = s.raw_scores(term)
    if learned:
        raw, _ = s.learner.blend(term, raw)
    return (raw - raw.mean()) / (raw.std() + 1e-9)


def visual_terms():
    """Lexicon terms to tag by look, keyed by display spelling. Skips family words
    (those come from linked people) and duplicate spellings (ensemada -> ensaymada)."""
    raw = yaml.safe_load((CONFIG / "lexicon.yaml").read_text()) or {}
    kin = {norm(k) for k in (yaml.safe_load((CONFIG / "kinship.yaml").read_text()) or {})} \
        if (CONFIG / "kinship.yaml").exists() else set()
    seen, out = set(), {}
    for key, phrases in raw.items():
        n = norm(key)
        sig = tuple(phrases or [])
        if n in kin or sig in seen:
            continue
        seen.add(sig)
        out[str(key)] = n
    return out


def propose(s: Searcher, z_cut: float) -> dict:
    """term -> list of uuids proposed for that keyword."""
    ids = s.meta.uuid.to_numpy()
    proposals = {}
    # 1) family words: photos that contain the linked people (certain, no threshold)
    for term, people in s.culture.kin.items():
        if people:
            hit = s._has_any(people)
            if hit.any():
                proposals[term] = list(ids[hit])
    # 2) everything else: by how it looks
    for display, term in visual_terms().items():
        z = zscores(s, term)
        top = np.argsort(-z)[:MAX_PER_TERM]
        top = top[z[top] >= z_cut]
        if len(top):
            proposals[display] = list(ids[top])
    return proposals


def calibrate(s: Searcher, target=0.85, pool=20):
    """Use your dev labels to choose the cutoff.

    Only photos you actually looked at while labeling (each method's top 20) are
    scored, so a correct photo you never saw doesn't count as a wrong tag.
    """
    labels_path = DATA / "labels.json"
    if not labels_path.exists():
        sys.exit("No data/labels.json yet. Run label.py and eval.py first.")
    labels = json.loads(labels_path.read_text()).get("labels", {})
    terms = set(visual_terms().values())
    ids = s.meta.uuid.to_numpy()
    rows = []
    for q, rel in labels.items():
        if norm(q) in terms and rel:
            seen = set()
            for base in (True, False):  # rebuild the pool you labeled from
                res, _ = s.search(q, k=pool, baseline=base, learned=False)
                seen |= set(res.uuid)
            rows.append((zscores(s, norm(q), learned=False), np.isin(ids, list(rel)), np.isin(ids, list(seen))))
    if not rows:
        sys.exit("None of your labeled queries are single lexicon words; can't calibrate.")
    print(f"Calibrating on {len(rows)} labeled food/thing words (only photos you reviewed)\n")
    print("cutoff  precision  recall    F1   tags")
    results = []
    for cut in np.arange(1.0, 5.01, 0.25):
        tp = fp = fn = 0
        for z, rel, seen in rows:
            pred = np.zeros_like(rel)
            top = np.argsort(-z)[:MAX_PER_TERM]
            pred[top[z[top] >= cut]] = True
            pred &= seen
            tp += (pred & rel).sum(); fp += (pred & ~rel).sum(); fn += (~pred & rel).sum()
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        results.append((float(cut), p, r, f1, tp))
        print(f"{cut:5.2f}   {p:8.2f}  {r:6.2f}  {f1:5.2f}  {tp + fp:5}")
    hit = [c for c, p, r, f1, tp in results if p >= target and tp]
    if hit:
        best, why = hit[0], f"lowest cutoff with >= {target:.0%} precision"
    else:
        best = max(results, key=lambda x: x[3])[0]
        why = (f"no cutoff reached {target:.0%} precision, so using the best balance (F1). "
               "The review page is where you remove the misses")
    THRESHOLD_FILE.write_text(json.dumps({"z": best}))
    print(f"\nUsing cutoff {best:.2f}: {why}.")


def review_page(s: Searcher, proposals: dict):
    by_uuid = s.meta.set_index("uuid")
    thumbs = {u: thumb_data_uri(by_uuid.at[u, "image_path"], size=200)
              for ids in proposals.values() for u in ids}
    page = DATA / "tag_review.html"
    page.write_text(REVIEW.replace("__DATA__", json.dumps({"proposals": proposals, "thumbs": thumbs})))
    total = sum(map(len, proposals.values()))
    print(f"Proposed {total} tags across {len(proposals)} keywords. Review: {page}")
    if sys.platform == "darwin":
        subprocess.run(["open", str(page)])


def _photos():
    try:
        import photoscript
    except ImportError:
        raise RuntimeError("photoscript isn't installed (it comes with osxphotos on a Mac).")
    photoscript.PhotosLibrary()  # launches Photos if needed
    return photoscript


def write_tags(approved: dict, progress=True) -> tuple[int, int]:
    """approved: keyword -> list of uuids. Returns (photos tagged, photos not found).
    Also used by the app's "Tag in Photos" button."""
    ps = _photos()
    by_photo: dict[str, set] = {}
    for kw, ids in approved.items():
        for u in ids:
            by_photo.setdefault(u, set()).add(kw)
    applied = json.loads(APPLIED.read_text()) if APPLIED.exists() else {}
    items = by_photo.items()
    if progress:
        from tqdm import tqdm
        items = tqdm(items, desc="Tagging")
    missing = 0
    for u, kws in items:
        try:
            photo = ps.Photo(u)
        except ValueError:
            missing += 1
            continue
        current = list(photo.keywords)
        added = [k for k in sorted(kws | {MARKER}) if k not in current]
        if added:
            photo.keywords = current + added
            applied[u] = sorted(set(applied.get(u, [])) | set(added))
    APPLIED.write_text(json.dumps(applied, indent=1))
    return len(by_photo) - missing, missing


def apply(path: Path):
    tagged, missing = write_tags(json.loads(path.read_text()))
    print(f"Tagged {tagged} photos.")
    if missing:
        print(f"{missing} photos weren't found. Is Photos open on the same library you indexed?")
    print('Try it: search "halo-halo" in the Photos app.')


def remove_tags() -> int:
    """Remove every keyword Alaala added. Returns number of photos touched."""
    if not APPLIED.exists():
        return 0
    ps = _photos()
    applied = json.loads(APPLIED.read_text())
    for u, kws in applied.items():
        try:
            photo = ps.Photo(u)
        except ValueError:
            continue
        photo.keywords = [k for k in photo.keywords if k not in kws]
    APPLIED.unlink()
    return len(applied)


def undo():
    n = remove_tags()
    print(f"Removed Alaala's keywords from {n} photos." if n else "Nothing to undo.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--apply", nargs="?", const="", metavar="TAGS_JSON",
                    help="approved tags file (default: newest tags*.json in Downloads)")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--z", type=float, help="override the cutoff")
    args = ap.parse_args()

    if args.undo:
        return undo()
    if args.apply is not None:
        path = Path(args.apply).expanduser() if args.apply else max(
            Path("~/Downloads").expanduser().glob("tags*.json"), key=lambda p: p.stat().st_mtime, default=None)
        if not path or not path.exists():
            sys.exit("Couldn't find tags.json. Click 'Save approved tags' on the review page first.")
        print(f"Applying {path}")
        return apply(path)

    s = Searcher()
    if args.calibrate:
        return calibrate(s)
    z = args.z or (json.loads(THRESHOLD_FILE.read_text())["z"] if THRESHOLD_FILE.exists() else DEFAULT_Z)
    print(f"Cutoff: {z:.2f}")
    review_page(s, propose(s, z))


REVIEW = r"""<!doctype html><meta charset=utf-8><title>Alaala: review tags</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:12px 24px;display:flex;gap:16px;align-items:center;z-index:9;border-bottom:1px solid #333}
header b{font-size:16px} button{font:inherit;padding:6px 14px;border-radius:6px;border:0;cursor:pointer}
.save{background:#22c55e;color:#062} .all{background:#333;color:#ccc}
section{padding:16px 24px;border-bottom:1px solid #222}
h2{margin:0;display:flex;gap:12px;align-items:center} h2 small{opacity:.6;font-weight:400;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:10px}
img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer;border:3px solid #22c55e}
img.off{border-color:transparent;opacity:.3;filter:grayscale(1)}
</style>
<header><b>Review tags</b><span id=tot></span><span style="flex:1"></span>
<span style="opacity:.6">Green = will be tagged. Click to remove wrong ones.</span>
<button class=save onclick="save()">Save approved tags</button></header>
<div id=app></div>
<script>
const D = __DATA__;
let off = {};
try { off = JSON.parse(localStorage.getItem("alaala-tag-off")) || {}; } catch (e) {}
const persist = () => { try { localStorage.setItem("alaala-tag-off", JSON.stringify(off)); } catch (e) {} tot(); };
const kept = kw => D.proposals[kw].filter(u => !(off[kw] || []).includes(u));
const tot = () => { document.getElementById("tot").textContent =
  Object.keys(D.proposals).reduce((n, k) => n + kept(k).length, 0) + " tags approved"; };
for (const [kw, ids] of Object.entries(D.proposals)) {
  const sec = document.createElement("section");
  sec.innerHTML = `<h2>${kw}<small></small><button class=all>Skip this keyword</button></h2><div class=grid></div>`;
  const small = sec.querySelector("small"), grid = sec.querySelector(".grid");
  const refresh = () => small.textContent = kept(kw).length + " / " + ids.length + " photos";
  sec.querySelector(".all").onclick = () => { off[kw] = kept(kw).length ? [...ids] : [];
    grid.querySelectorAll("img").forEach(im => im.classList.toggle("off", off[kw].length > 0)); refresh(); persist(); };
  ids.forEach(u => {
    const im = document.createElement("img"); im.src = D.thumbs[u]; im.loading = "lazy";
    if ((off[kw] || []).includes(u)) im.classList.add("off");
    im.onclick = () => { off[kw] = off[kw] || []; const i = off[kw].indexOf(u);
      i >= 0 ? off[kw].splice(i, 1) : off[kw].push(u); im.classList.toggle("off"); refresh(); persist(); };
    grid.appendChild(im);
  });
  refresh(); document.getElementById("app").appendChild(sec);
}
tot();
function save() {
  const out = {}; for (const kw of Object.keys(D.proposals)) { const k = kept(kw); if (k.length) out[kw] = k; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "tags.json"; a.click();
}
</script>"""

if __name__ == "__main__":
    main()
