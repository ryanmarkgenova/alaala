"""Alaala learns from you.

Every "yes, this is halo-halo" and "no, it isn't" is saved. Once a search has a few
yeses, Alaala trains a small model for it (logistic regression on the photo vectors,
a.k.a. a linear probe) and blends it with the zero-shot score. The more examples,
the more it trusts what it learned:  trust = examples / (examples + 8).

    python learn.py --seed      # turn your dev labels into training examples
    python learn.py --train     # retrain every search you've taught it
    python learn.py --stats     # what it has learned so far
    python learn.py --curve     # learning curve chart: does it improve with examples?

The held-out TEST labels are never used here, so eval.py's test numbers stay honest.
"""
import argparse
import json
import sqlite3
import threading
import time
import zlib

import numpy as np

from common import DATA, ROOT
from culture import norm

DB = DATA / "feedback.sqlite"
PROBES = DATA / "probes.json"
MIN_POS = 3          # examples needed before a search gets its own model
PRIOR = 8            # trust = n / (n + PRIOR)
N_BACKGROUND = 200   # unlabeled photos used as "probably not" examples
BG_WEIGHT = 0.3      # ...counted at 30% weight, since some might actually match


class Feedback:
    """SQLite store of every yes/no answer. Latest answer for a photo wins."""

    def __init__(self, path=DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute("""CREATE TABLE IF NOT EXISTS feedback (
                key TEXT, uuid TEXT, label INTEGER, source TEXT, ts REAL,
                PRIMARY KEY (key, uuid))""")
            self.con.commit()

    def add(self, key, uuid, label, source):
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?)",
                             (norm(key), uuid, int(label), source, time.time()))
            self.con.commit()

    def labels(self, key) -> dict:
        with self.lock:
            rows = self.con.execute("SELECT uuid, label FROM feedback WHERE key = ?", (norm(key),)).fetchall()
        return dict(rows)

    def keys(self):
        with self.lock:
            return self.con.execute("""SELECT key, SUM(label), COUNT(*) - SUM(label)
                                       FROM feedback GROUP BY key ORDER BY SUM(label) DESC""").fetchall()


def train_probe(X, pos, neg, bg):
    """Logistic regression: yes photos vs no photos (+ down-weighted background)."""
    from sklearn.linear_model import LogisticRegression

    idx = np.concatenate([pos, neg, bg]).astype(int)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg) + len(bg))]
    w = np.r_[np.ones(len(pos) + len(neg)), np.full(len(bg), BG_WEIGHT)]
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    clf.fit(X[idx], y, sample_weight=w)
    return clf.coef_[0], float(clf.intercept_[0])


def background(n_total, exclude, key, n=N_BACKGROUND):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    pool = np.setdiff1d(np.arange(n_total), np.asarray(list(exclude), dtype=int))
    return rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else np.array([], int)


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def blend_scores(raw, logit, n_pos):
    """Mix zero-shot and learned scores in z-space, then map back to the raw scale
    (so the person boost and everything downstream behave the same)."""
    a = n_pos / (n_pos + PRIOR)
    z = (1 - a) * zs(raw) + a * zs(logit)
    return z * raw.std() + raw.mean(), a


class Learner:
    def __init__(self, vecs: np.ndarray, uuids):
        self.X = vecs
        self.ids = list(uuids)
        self.row = {u: i for i, u in enumerate(self.ids)}
        self.fb = Feedback()
        self.probes = json.loads(PROBES.read_text()) if PROBES.exists() else {}

    def _save(self):
        PROBES.write_text(json.dumps(self.probes))

    def stats(self, key):
        lab = self.fb.labels(key)
        p = self.probes.get(norm(key))
        n_pos = sum(lab.values())
        return {"key": norm(key), "yes": n_pos, "no": len(lab) - n_pos, "trained": bool(p),
                "trust": round(p["n_pos"] / (p["n_pos"] + PRIOR), 2) if p else 0.0,
                "needed": max(0, MIN_POS - n_pos)}

    def record(self, key, uuid, label, source="app"):
        if uuid in self.row:
            self.fb.add(key, uuid, label, source)
        return self.train(key)

    def train(self, key):
        key = norm(key)
        lab = self.fb.labels(key)
        pos = [self.row[u] for u, l in lab.items() if l == 1 and u in self.row]
        neg = [self.row[u] for u, l in lab.items() if l == 0 and u in self.row]
        if len(pos) < MIN_POS:
            self.probes.pop(key, None)
        else:
            bg = background(len(self.ids), set(pos) | set(neg), key)
            w, b = train_probe(self.X, np.array(pos), np.array(neg, int), bg)
            self.probes[key] = {"w": w.round(5).tolist(), "b": b, "n_pos": len(pos), "n_neg": len(neg)}
        self._save()
        return self.stats(key)

    def train_all(self):
        return [self.train(k) for k, _, _ in self.fb.keys()]

    def logit(self, key):
        p = self.probes.get(norm(key))
        return None if p is None else self.X @ np.asarray(p["w"], dtype=np.float32) + p["b"]

    def blend(self, key, raw):
        p = self.probes.get(norm(key))
        if p is None:
            return raw, None
        out, a = blend_scores(raw, self.logit(key), p["n_pos"])
        return out, {"examples": p["n_pos"], "trust": round(a, 2)}

    def candidates(self, key, raw, n=12):
        """Active learning: what should we ask about next?
        No model yet: the top unlabeled matches (fastest way to collect yeses).
        Model trained: the photos it's least sure about (most informative)."""
        lab = self.fb.labels(key)
        labeled = np.array([u in lab for u in self.ids])
        scores, _ = self.blend(key, raw)
        order = [i for i in np.argsort(-scores) if not labeled[i]]
        lg = self.logit(key)
        if lg is None:
            return order[:n]
        prob = 1 / (1 + np.exp(-lg))
        shortlist = order[:300]
        return sorted(shortlist, key=lambda i: abs(prob[i] - 0.5))[:n]


# ---------------------------------------------------------------- CLI helpers

def labeled_pool(s, q, k=20):
    seen = set()
    for base in (True, False):
        res, _ = s.search(q, k=k, baseline=base, learned=False)
        seen |= set(res.uuid)
    return seen


def seed(s):
    """Dev labels -> feedback. Picked photos are yes; photos you saw but didn't pick are no."""
    path = DATA / "labels.json"
    if not path.exists():
        raise SystemExit("No data/labels.json (dev labels) found. Run label.py and eval.py first.")
    data = json.loads(path.read_text())
    n = 0
    for q, rel in data.get("labels", {}).items():
        rel = set(rel)
        for u in labeled_pool(s, q) | rel:
            s.learner.fb.add(q, u, int(u in rel), "eval-dev")
            n += 1
    for q in data.get("none", []):
        for u in labeled_pool(s, q):
            s.learner.fb.add(q, u, 0, "eval-dev")
            n += 1
    print(f"Imported {n} answers from your dev labels.")
    for st in s.learner.train_all():
        if st["trained"]:
            print(f"  {st['key']:16} learned from {st['yes']} yes / {st['no']} no  (trust {st['trust']})")


def ndcg10(scores, rel_idx):
    top = np.argsort(-scores)[:10]
    rel = set(rel_idx)
    dcg = sum(1 / np.log2(i + 2) for i, j in enumerate(top) if j in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    return dcg / idcg if idcg else 0.0


def curve(s, ks=(0, 1, 2, 3, 5, 8, 12), seeds=10, min_test=3, min_searches=5):
    """For each taught search: hide some yes photos, train on k others, and measure
    how well the hidden ones are found.

    Fair comparison: at every k, learned is compared with plain zero-shot Alaala on the
    SAME searches and the same hidden photos (paired), because searches with many
    examples may simply be easier.
    """
    import pandas as pd

    X, L = s.vecs, s.learner
    rows = []
    for key, n_pos, n_neg in L.fb.keys():
        lab = L.fb.labels(key)
        pos = np.array([L.row[u] for u, l in lab.items() if l == 1 and u in L.row])
        neg = np.array([L.row[u] for u, l in lab.items() if l == 0 and u in L.row], dtype=int)
        if len(pos) < min_test + 1:
            continue
        raw = s.raw_scores(key)
        for k in ks:
            if k == 0 or k > len(pos) - min_test:
                continue
            learned, zero = [], []
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                perm = rng.permutation(len(pos))
                tr, te = pos[perm[:k]], pos[perm[k:]]
                tr_neg = rng.permutation(neg)[: len(neg) // 2] if len(neg) else neg
                hide = np.r_[tr, tr_neg].astype(int)   # don't reward re-finding training photos
                bg = background(len(X), set(tr) | set(tr_neg), key + str(sd))
                w, b = train_probe(X, tr, tr_neg, bg)
                sc, _ = blend_scores(raw, X @ w + b, k)
                sc[hide] = -np.inf
                z = raw.copy()
                z[hide] = -np.inf
                learned.append(ndcg10(sc, te))
                zero.append(ndcg10(z, te))
            rows.append({"search": key, "examples": k, "learned": float(np.mean(learned)),
                         "zero_shot": float(np.mean(zero))})
    if not rows:
        raise SystemExit("Not enough taught searches yet (need ones with 4+ yes photos). Run --seed or teach more.")
    df = pd.DataFrame(rows)
    df["gain"] = df.learned - df.zero_shot
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "learning_curve.csv", index=False)
    summary = (df.groupby("examples")
                 .agg(zero_shot=("zero_shot", "mean"), learned=("learned", "mean"), gain=("gain", "mean"),
                      wins=("gain", lambda g: int((g > 0.005).sum())), searches=("search", "nunique"))
                 .reset_index())
    summary = summary[summary.searches >= min_searches]
    print("Paired: learned vs zero-shot on the same searches and the same hidden photos\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if summary.empty:
        raise SystemExit(f"No example count has {min_searches}+ searches yet. Teach more.")
    plot_curve(summary, ROOT / "docs" / "learning_curve.png")
    return summary


def plot_curve(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, blue = "#fcfcfb", "#1f1b16", "#6b645c", "#e8e5df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    x = summary.examples
    ax.plot(x, summary.zero_shot, color=muted, lw=1.5, ls=(0, (4, 4)), marker="o", ms=6,
            markerfacecolor=surface, markeredgecolor=muted, zorder=2)
    ax.plot(x, summary.learned, color=blue, lw=2, marker="o", ms=8,
            markeredgecolor=surface, markeredgewidth=2, zorder=3)
    last = summary.iloc[-1]
    up = last.learned >= last.zero_shot
    ax.annotate(f"Learned  {last.learned:.2f}", (last.examples, last.learned), xytext=(10, 6 if up else -6),
                textcoords="offset points", va="center", color=ink, fontsize=10, fontweight="bold")
    ax.annotate(f"Zero-shot, same searches  {last.zero_shot:.2f}", (last.examples, last.zero_shot),
                xytext=(10, -6 if up else 6), textcoords="offset points", va="center", color=muted, fontsize=9.5)
    for _, r in summary.iterrows():  # how many searches each point averages
        ax.annotate(f"n={int(r.searches)}", (r.examples, 0.03), ha="center", color=muted, fontsize=8)
    ax.set_xlabel("Photos you confirmed for that search", color=muted, fontsize=10)
    ax.set_ylabel("nDCG@10 on held-out photos", color=muted, fontsize=10)
    best = summary.loc[summary.gain.idxmax()]
    title = (f"Teaching Alaala: +{best.gain:.2f} nDCG over zero-shot at {int(best.examples)} examples"
             if best.gain > 0.01 else "Teaching Alaala: no clear gain yet")  # title states what the data shows
    ax.set_title(title, loc="left", color=ink, fontsize=12.5, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=grid, lw=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9, length=0)
    ax.set_xlim(x.min() - 0.6, x.max() + (x.max() - x.min()) * 0.42)  # room for end labels
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, facecolor=surface)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--curve", action="store_true")
    args = ap.parse_args()

    from search import Searcher

    s = Searcher()
    if args.seed:
        seed(s)
    if args.train:
        for st in s.learner.train_all():
            print(st)
    if args.stats or not any(vars(args).values()):
        for key, y, n in s.learner.fb.keys():
            st = s.learner.stats(key)
            print(f"{key:20} {y:3} yes {n:4} no   " + (f"model trained (trust {st['trust']})" if st["trained"]
                                                      else f"needs {st['needed']} more yes"))
    if args.curve:
        curve(s)


if __name__ == "__main__":
    main()
