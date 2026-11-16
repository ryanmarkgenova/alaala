"""Search your library.

    python search.py "lolo singing videoke"           # culture-aware search (default)
    python search.py "lolo" --baseline                # plain CLIP, for comparison
    python search.py "lolo" --compare                 # side by side: baseline vs culture
    python search.py "birthday" --person "Exact Name" --year 2019
    python search.py --people                         # list everyone named in Photos
    python search.py                                  # interactive mode

Results print in the terminal and open as a thumbnail grid in your browser.
"""
import argparse
import html
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

from common import DATA, EMB, EMB_IDS, INDEX, load_model
from culture import Culture

PERSON_BOOST = 0.10  # CLIP scores sit around 0.2-0.35, so this lifts linked people to the top
KEEP_ORIGINAL = True  # v2: tied on dev, won on the held-out test set
FOUND_FILE = DATA / "found_threshold.json"
DEFAULT_FOUND_Z = 3.5  # replaced by `python eval.py --found`


class Searcher:
    def __init__(self):
        self.model, _, self.tokenizer, self.dev = load_model()
        self.vecs = np.load(EMB).astype(np.float32)
        ids = np.load(EMB_IDS, allow_pickle=True)
        meta = pd.read_parquet(INDEX).set_index("uuid")
        self.meta = meta.loc[ids].reset_index()  # row-aligned with vecs
        day = self.meta["date_local"] if "date_local" in self.meta else self.meta.date.dt.strftime("%Y-%m-%d")
        self.mmdd = day.str[5:].to_numpy()
        self.names = Counter(n for ps in self.meta.persons for n in ps)
        self.culture = Culture(list(self.names))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embed one or more phrasings and average them (prompt ensemble)."""
        with torch.no_grad():
            t = self.tokenizer(texts).to(self.dev)
            f = self.model.encode_text(t)
            f = f / f.norm(dim=-1, keepdim=True)
        v = f.float().cpu().numpy().mean(axis=0)
        return v / np.linalg.norm(v)

    def confidence(self, query: str) -> float:
        """How strongly the best photo stands out from the rest of the library
        (z-score of the top match). Low means "probably not in your photos"."""
        parsed = self.culture.parse(query, keep_original=KEEP_ORIGINAL)
        raw = self.vecs @ self.encode_texts(parsed.variants)
        return float(((raw - raw.mean()) / (raw.std() + 1e-9)).max())

    def found_threshold(self) -> float:
        import json
        return json.loads(FOUND_FILE.read_text())["z"] if FOUND_FILE.exists() else DEFAULT_FOUND_Z

    def assess(self, query: str, parsed) -> dict:
        """Is the thing actually in the library? If not, suggest nearby searches that are."""
        if parsed and parsed.persons:  # linked family members: definitely there
            return {"found": True, "confidence": None, "similar": []}
        conf = self.confidence(query)
        cut = self.found_threshold()
        if conf >= cut:
            return {"found": True, "confidence": round(conf, 2), "similar": []}
        return {"found": False, "confidence": round(conf, 2), "similar": self.similar_terms(query, cut)}

    def similar_terms(self, query: str, cut: float, n: int = 4) -> list[str]:
        """Lexicon words that mean something close to the query AND exist in this library."""
        if not hasattr(self, "_term_vecs"):
            terms, seen = [], set()
            for term, phrases in self.culture.lexicon.items():
                sig = tuple(phrases or [])
                if sig not in seen and term not in self.culture.kin:
                    seen.add(sig)
                    terms.append(term)
            self._terms = terms
            self._term_vecs = np.stack([self.encode_texts(self.culture.lexicon[t]) for t in terms])
            self._term_conf = {}
        own = set(self.culture.parse(query).terms)
        q = self.encode_texts(self.culture.parse(query, keep_original=KEEP_ORIGINAL).variants)
        out = []
        for i in np.argsort(-(self._term_vecs @ q)):
            t = self._terms[i]
            if t in own:
                continue
            if t not in self._term_conf:
                self._term_conf[t] = self.confidence(t)
            if self._term_conf[t] >= cut:
                out.append(t)
            if len(out) == n:
                break
        return out

    def _has_any(self, people: set[str]) -> np.ndarray:
        return self.meta.persons.apply(lambda ps: bool(people & set(ps))).to_numpy()

    def _in_window(self, start: str, end: str) -> np.ndarray:
        if start <= end:
            return (self.mmdd >= start) & (self.mmdd <= end)
        return (self.mmdd >= start) | (self.mmdd <= end)  # wraps new year

    def search(self, query, k=24, baseline=False, persons=None, year=None, videos=None,
               keep_original=None):
        parsed = None
        if baseline:
            scores = self.vecs @ self.encode_texts([query])
        else:
            keep = KEEP_ORIGINAL if keep_original is None else keep_original
            parsed = self.culture.parse(query, keep_original=keep)
            scores = self.vecs @ self.encode_texts(parsed.variants)
            if parsed.persons:
                scores = scores + PERSON_BOOST * self._has_any(parsed.persons)

        m = self.meta
        mask = np.ones(len(scores), dtype=bool)
        if parsed and parsed.date_windows:
            win = np.zeros(len(scores), dtype=bool)
            for s, e in parsed.date_windows:
                win |= self._in_window(s, e)
            if win.any():
                mask &= win
            else:
                parsed.notes.append("(no photos in that date window, ignoring it)")
        if persons:
            want = set(persons)
            mask &= m.persons.apply(lambda ps: want.issubset(set(ps))).to_numpy()
        if year:
            mask &= (m.date.dt.year == year).to_numpy()
        if videos is not None:
            mask &= (m.is_video == videos).to_numpy()

        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        top = top[np.isfinite(scores[top])]
        out = m.iloc[top].copy()
        out["score"] = scores[top]
        return out, parsed


def print_results(df):
    for r in df.itertuples():
        who = ", ".join(r.persons) or "-"
        kind = "video" if r.is_video else "photo"
        print(f"{r.score:.3f}  {r.date:%Y-%m-%d}  {kind:5}  {who[:40]:40}  {r.place or ''}")


def thumb_data_uri(path, size=400):
    """Embed a small JPEG in the page. Browsers can't read inside a .photoslibrary
    (macOS privacy) or render HEIC, so Python does the reading and converting."""
    import base64
    import io

    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _grid(df):
    return "".join(
        f'<figure><img src="{thumb_data_uri(r.image_path)}" loading="lazy">'
        f"<figcaption>{r.score:.3f} · {r.date:%Y-%m-%d}<br>"
        f'{html.escape(", ".join(r.persons))}</figcaption></figure>'
        for r in df.itertuples()
    )


def write_html(query, sections, path=DATA / "results.html"):
    """sections: list of (label, dataframe). Two sections render side by side."""
    cols = len(sections)
    body = "".join(
        f"<section><h3>{html.escape(label)}</h3><main>{_grid(df)}</main></section>"
        for label, df in sections
    )
    path.write_text(
        f"""<!doctype html><meta charset=utf-8><title>{html.escape(query)}</title>
<style>body{{font:14px system-ui;margin:24px;background:#111;color:#eee}}
.wrap{{display:grid;grid-template-columns:repeat({cols},1fr);gap:32px}}
h3{{opacity:.7;font-weight:500}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax({200 if cols == 1 else 140}px,1fr));gap:12px}}
img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px}}
figure{{margin:0}}figcaption{{font-size:12px;opacity:.7;margin-top:4px}}</style>
<h2>{html.escape(query)}</h2><div class=wrap>{body}</div>"""
    )
    return path


def run(s, args, query):
    kw = dict(k=args.k, persons=args.person, year=args.year, videos=True if args.videos else None)
    if args.compare:
        base, _ = s.search(query, baseline=True, **kw)
        cult, parsed = s.search(query, **kw)
        print(parsed.explain())
        sections = [("Plain search (baseline)", base), ("Alaala (culture-aware)", cult)]
        res = cult
    else:
        res, parsed = s.search(query, baseline=args.baseline, **kw)
        if parsed:
            print(parsed.explain())
            a = s.assess(query, parsed)
            if not a["found"]:
                print(f"  No {query} found in your photos. Closest matches below."
                      + (f" Try: {', '.join(a['similar'])}" if a["similar"] else ""))
        sections = [("Baseline" if args.baseline else "Alaala", res)]
    if res.empty:
        print("No results (check your filters)")
        return
    print_results(res)
    if not args.no_html:
        page = write_html(query, sections)
        if sys.platform == "darwin":
            subprocess.run(["open", str(page)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--baseline", action="store_true", help="plain CLIP, no culture layer")
    ap.add_argument("--compare", action="store_true", help="baseline vs culture side by side")
    ap.add_argument("--person", action="append", help="name as it appears in Photos > People")
    ap.add_argument("--year", type=int)
    ap.add_argument("--videos", action="store_true", help="only videos")
    ap.add_argument("--people", action="store_true", help="list named people and exit")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    s = Searcher()
    if args.people:
        for name, n in s.names.most_common():
            linked = [t for t, names in s.culture.kin.items() if name in names]
            print(f"{n:5}  {name}" + (f"   (linked: {', '.join(linked)})" if linked else ""))
        return
    if args.query:
        run(s, args, args.query)
        return
    print("Type a search (blank line to quit)")
    while q := input("> ").strip():
        run(s, args, q)


if __name__ == "__main__":
    main()
