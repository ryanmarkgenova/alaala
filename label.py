"""Build the labeling page for the eval.

For each query in eval/queries.txt, it pools the top results from BOTH the
baseline and Alaala, shuffles them by date (so you can't tell which method
found what), and lets you click every photo that truly matches the query.

    python label.py                                   # dev set (eval/queries.txt)
    python label.py --queries eval/test_queries.txt --name labels_test   # held-out test set

Click photos to mark them correct. If nothing matches a query, click
"No matches". When done, hit "Save labels" and it downloads labels.json.
Your clicks are also kept in the browser, so you can close it and come back.
"""
import json
from pathlib import Path

from common import DATA, ROOT
from search import Searcher, thumb_data_uri

QUERIES = ROOT / "eval" / "queries.txt"
LABELS = DATA / "labels.json"
POOL = 20  # top N from each method


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(QUERIES))
    ap.add_argument("--name", default="labels", help="labels file name (and download name)")
    args = ap.parse_args()
    labels_path = DATA / f"{args.name}.json"

    s = Searcher()
    queries = [
        line.strip()
        for line in Path(args.queries).expanduser().read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    by_uuid = s.meta.set_index("uuid")
    pools, thumbs = {}, {}
    for q in queries:
        base, _ = s.search(q, k=POOL, baseline=True)
        cult, _ = s.search(q, k=POOL, learned=False)  # neutral pool for fair labeling
        ids = list(dict.fromkeys(list(base.uuid) + list(cult.uuid)))
        ids.sort(key=lambda u: by_uuid.at[u, "date"])  # blind: order by date, not score
        pools[q] = ids
        for u in ids:
            if u not in thumbs:
                thumbs[u] = thumb_data_uri(by_uuid.at[u, "image_path"], size=220)
        print(f"{q:20} {len(ids)} candidates")

    existing = json.loads(labels_path.read_text()) if labels_path.exists() else {"labels": {}, "none": []}
    page = DATA / f"{args.name}.html"
    page.write_text(TEMPLATE.replace("__DATA__", json.dumps(
        {"queries": queries, "pools": pools, "thumbs": thumbs, "existing": existing,
         "name": args.name}
    )))
    print(f"\nOpen {page}")
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", str(page)])


TEMPLATE = r"""<!doctype html><meta charset=utf-8><title>Alaala labeling</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:12px 24px;display:flex;gap:16px;align-items:center;z-index:9;border-bottom:1px solid #333}
header b{font-size:16px} button{font:inherit;padding:6px 14px;border-radius:6px;border:0;cursor:pointer}
.save{background:#3b82f6;color:#fff} .none{background:#333;color:#ccc} .none.on{background:#b45309;color:#fff}
section{padding:16px 24px;border-bottom:1px solid #222}
h2{margin:0 0 4px;display:flex;gap:12px;align-items:center} h2 small{opacity:.6;font-weight:400;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:10px}
img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer;border:3px solid transparent;opacity:.75}
img.on{border-color:#22c55e;opacity:1}
</style>
<header><b>Alaala labeling</b><span id=prog></span><span style="flex:1"></span>
<span style="opacity:.6">Click every photo that truly matches the search</span>
<button class=save onclick="save()">Save labels</button></header>
<div id=app></div>
<script>
const D = __DATA__;
let st = {labels: {}, none: []};
try { st = JSON.parse(localStorage.getItem("alaala-" + D.name)) || D.existing; } catch (e) { st = D.existing; }
st.labels = st.labels || {}; st.none = st.none || [];
// keep only this set's queries
for (const q of Object.keys(st.labels)) if (!D.queries.includes(q)) delete st.labels[q];
st.none = st.none.filter(q => D.queries.includes(q));
const persist = () => { try { localStorage.setItem("alaala-" + D.name, JSON.stringify(st)); } catch (e) {} prog(); };
const done = q => (st.labels[q] || []).length > 0 || st.none.includes(q);
const prog = () => { document.getElementById("prog").textContent =
  D.queries.filter(done).length + " / " + D.queries.length + " queries labeled"; };
const app = document.getElementById("app");
D.queries.forEach(q => {
  const sec = document.createElement("section");
  const sel = new Set(st.labels[q] || []);
  sec.innerHTML = `<h2>${q}<small></small><button class="none">No matches</button></h2><div class=grid></div>`;
  const count = sec.querySelector("small"), none = sec.querySelector(".none"), grid = sec.querySelector(".grid");
  const refresh = () => { count.textContent = sel.size + " selected";
    none.classList.toggle("on", st.none.includes(q)); };
  none.onclick = () => { const i = st.none.indexOf(q);
    if (i >= 0) st.none.splice(i, 1); else { st.none.push(q); sel.clear();
      grid.querySelectorAll("img").forEach(im => im.classList.remove("on")); st.labels[q] = []; }
    refresh(); persist(); };
  D.pools[q].forEach(u => {
    const im = document.createElement("img"); im.src = D.thumbs[u]; im.loading = "lazy";
    if (sel.has(u)) im.classList.add("on");
    im.onclick = () => { sel.has(u) ? sel.delete(u) : sel.add(u); im.classList.toggle("on");
      st.labels[q] = [...sel]; const i = st.none.indexOf(q); if (i >= 0) st.none.splice(i, 1);
      refresh(); persist(); };
    grid.appendChild(im);
  });
  refresh(); app.appendChild(sec);
});
prog();
function save() {
  const blob = new Blob([JSON.stringify(st, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = D.name + ".json"; a.click();
}
</script>"""

if __name__ == "__main__":
    main()
