"""Step 2: turn every photo into a vector with MobileCLIP2.

Resumable: if it stops (laptop sleeps, you Ctrl+C), just run it again and it
picks up where it left off.

    python embed.py
    python embed.py --batch 16    # lower this if you see memory pressure
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import CHUNKS, EMB, EMB_IDS, INDEX, load_model

try:  # originals can be HEIC
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def load_image(path, preprocess):
    try:
        with Image.open(path) as im:
            return preprocess(im.convert("RGB"))
    except Exception:
        return None


SKIPPED = CHUNKS / "skipped.txt"  # unreadable files, so reruns don't retry them


def done_uuids() -> set[str]:
    done = set(SKIPPED.read_text().split()) if SKIPPED.exists() else set()
    for f in CHUNKS.glob("chunk_*.npz"):
        done.update(np.load(f, allow_pickle=True)["uuids"].tolist())
    return done


def merge_chunks():
    files = sorted(CHUNKS.glob("chunk_*.npz"))
    uuids, vecs = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        uuids.extend(z["uuids"].tolist())
        vecs.append(z["vecs"])
    np.save(EMB, np.concatenate(vecs).astype(np.float16))
    np.save(EMB_IDS, np.array(uuids, dtype=object), allow_pickle=True)
    print(f"Saved {len(uuids):,} embeddings to {EMB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024, help="save progress every N items")
    args = ap.parse_args()

    df = pd.read_parquet(INDEX)
    df = df[df.image_path.notna()]
    CHUNKS.mkdir(parents=True, exist_ok=True)
    done = done_uuids()
    todo = df[~df.uuid.isin(done)]
    print(f"{len(df):,} indexable, {len(done):,} already done, {len(todo):,} to go")

    if len(todo):
        model, preprocess, _, dev = load_model()
        print(f"Model loaded on {dev}")
        next_chunk = len(list(CHUNKS.glob("chunk_*.npz")))
        buf_ids, buf_vecs, skipped = [], [], 0
        items = list(zip(todo.uuid, todo.image_path))

        for i in tqdm(range(0, len(items), args.batch), desc="Embedding"):
            batch = items[i : i + args.batch]
            tensors, ids = [], []
            for uuid, path in batch:
                t = load_image(path, preprocess)
                if t is None:
                    skipped += 1
                    with open(SKIPPED, "a") as fh:
                        fh.write(uuid + "\n")
                    continue
                tensors.append(t)
                ids.append(uuid)
            if tensors:
                with torch.no_grad():
                    x = torch.stack(tensors).to(dev)
                    f = model.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                buf_vecs.append(f.float().cpu().numpy())
                buf_ids.extend(ids)

            if len(buf_ids) >= args.chunk or i + args.batch >= len(items):
                if buf_ids:
                    np.savez(
                        CHUNKS / f"chunk_{next_chunk:05d}.npz",
                        uuids=np.array(buf_ids, dtype=object),
                        vecs=np.concatenate(buf_vecs).astype(np.float16),
                    )
                    next_chunk += 1
                buf_ids, buf_vecs = [], []
        if skipped:
            print(f"Skipped {skipped} unreadable images")

    merge_chunks()


if __name__ == "__main__":
    main()
