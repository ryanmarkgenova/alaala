"""Score baseline vs Alaala on your labeled queries.

    python eval.py                     # finds labels.json in data/ or Downloads
    python eval.py --labels path/to/labels.json
    python eval.py --labels ~/Downloads/labels_test.json --out test   # held-out test set

Metrics (all @10, averaged over queries that have at least one correct photo):
  Hit@1      top result is correct
  Recall@10  share of the correct photos that show up in the top 10
  nDCG@10    rewards putting correct photos near the top
  MRR        1 / rank of the first correct photo

Labels come from pooling both methods' top 20 (standard "pooled" IR evaluation),
so recall is relative to what either method could find.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from common import DATA, ROOT
from search import Searcher

K = 10
METHODS = {  # name -> search() options. This is the ablation table.
    "Baseline (plain CLIP)": dict(baseline=True),
    "Alaala v1 (descriptions only)": dict(keep_original=False),
    "Alaala v2 (+ original word)": dict(keep_original=True),
}


def find_labels(arg):
    if arg:
        return Path(arg).expanduser()
    downloads = sorted((p for p in Path("~/Downloads").expanduser().glob("labels*.json")
                        if "test" not in p.name), key=lambda p: p.stat().st_mtime)
    local = DATA / "labels.json"
    if downloads and (not local.exists() or downloads[-1].stat().st_mtime > local.stat().st_mtime):
        shutil.copy(downloads[-1], local)  # keep a copy with the project
        print(f"Using {downloads[-1].name} from Downloads")
    return local


def md_table(df) -> str:
    """Markdown table without extra dependencies."""
    df = df.reset_index()
    cells = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in df.values]
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in cells])


def metrics(ranked, rel):
    top = ranked[:K]
    hits = [u in rel for u in top]
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), K)))
    first = next((i for i, u in enumerate(ranked) if u in rel), None)
    return {
        "Hit@1": float(hits[0]) if hits else 0.0,
        "Recall@10": sum(hits) / len(rel),
        "nDCG@10": dcg / idcg,
        "MRR": 1 / (first + 1) if first is not None else 0.0,
    }


def calibrate_found(data):
    """Pick the confidence cutoff that best separates searches that have matches
    from the ones you marked 'No matches' while labeling."""
    import numpy as np
    from search import FOUND_FILE

    s = Searcher()
    has = [q for q, v in data.get("labels", {}).items() if v and not s.culture.parse(q).persons]
    none = [q for q in data.get("none", []) if not s.culture.parse(q).persons]
    if not has or not none:
        raise SystemExit("Need labeled queries with matches AND some marked 'No matches'.")
    pos = {q: s.confidence(q) for q in has}
    neg = {q: s.confidence(q) for q in none}
    print("Confidence of the top match (z-score)\n")
    print("Marked 'No matches':", ", ".join(f"{q} {z:.2f}" for q, z in sorted(neg.items(), key=lambda x: -x[1])))
    print("Have matches:       ", ", ".join(f"{q} {z:.2f}" for q, z in sorted(pos.items(), key=lambda x: x[1])))
    best = None
    for cut in np.arange(1.5, 8.01, 0.05):
        tpr = np.mean([z >= cut for z in pos.values()])   # real matches we still show as found
        tnr = np.mean([z < cut for z in neg.values()])    # empty searches we correctly flag
        bal = (tpr + tnr) / 2
        if best is None or bal > best[1] + 1e-9:
            best = (float(cut), bal, tpr, tnr)
    cut, bal, tpr, tnr = best
    if bal <= 0.6:
        raise SystemExit(f"\nBest separation was only {bal:.0%} (coin flip is 50%), so keeping the default. "
                         "Label more searches, including a few 'No matches' ones, and try again.")
    FOUND_FILE.write_text(json.dumps({"z": round(cut, 2)}))
    print(f"\nCutoff {cut:.2f}: flags {tnr:.0%} of empty searches, keeps {tpr:.0%} of real ones as found "
          f"(balanced accuracy {bal:.0%}). Saved to {FOUND_FILE.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--out", default="dev", help="name for the saved report (dev or test)")
    ap.add_argument("--found", action="store_true",
                    help="calibrate the 'nothing found' cutoff from queries you marked No matches")
    args = ap.parse_args()
    if args.found:
        return calibrate_found(json.loads(find_labels(args.labels).read_text()))

    data = json.loads(find_labels(args.labels).read_text())
    labels = {q: set(v) for q, v in data.get("labels", {}).items() if v}
    skipped = data.get("none", [])

    s = Searcher()
    rows = []
    for q, rel in labels.items():
        kind = "Filipino term" if s.culture.parse(q).terms else "English"
        for name, opts in METHODS.items():
            res, _ = s.search(q, k=50, **opts)
            rows.append({"query": q, "type": kind, "method": name, **metrics(list(res.uuid), rel)})
    df = pd.DataFrame(rows)

    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"results_{args.out}.csv", index=False)

    cols = ["Hit@1", "Recall@10", "nDCG@10", "MRR"]
    overall = df.groupby("method")[cols].mean().reindex(METHODS)
    by_type = df.groupby(["type", "method"])[cols].mean()
    counts = df[df.method == list(METHODS)[0]].type.value_counts()

    piv = df.pivot(index="query", columns="method", values="nDCG@10")
    a, b = list(METHODS)[0], list(METHODS)[-1]  # baseline vs best version
    wins, ties, losses = (piv[b] > piv[a]).sum(), (piv[b] == piv[a]).sum(), (piv[b] < piv[a]).sum()

    fmt = md_table
    report = "\n".join([
        f"# Alaala eval ({args.out} set)\n",
        f"{len(labels)} labeled queries ({', '.join(f'{n} {t}' for t, n in counts.items())}); "
        f"{len(skipped)} marked as no matches and skipped.\n",
        "## Overall\n", fmt(overall), "",
        "## By query type\n", fmt(by_type), "",
        f"## Head to head (nDCG@10): {b} vs baseline\nWins {wins}, ties {ties}, loses {losses}.\n",
        "## Biggest gains vs baseline\n",
        fmt((piv[b] - piv[a]).sort_values(ascending=False).head(8).rename("nDCG gain").to_frame()), "",
        "## Where it still loses to baseline\n",
        fmt((piv[b] - piv[a]).sort_values().head(5).rename("nDCG change").to_frame()),
    ])
    (out / f"summary_{args.out}.md").write_text(report)
    print(report)
    print(f"\nSaved {out / f'summary_{args.out}.md'} and {out / f'results_{args.out}.csv'}")


if __name__ == "__main__":
    main()
