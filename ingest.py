"""Step 1: read your Apple Photos library into data/photos.parquet.

Works with "Optimize Mac Storage": uses the local preview images Photos keeps
on disk, and only falls back to the original when it's downloaded.

    python ingest.py                     # default system library
    python ingest.py --library "/path/to/Photos Library.photoslibrary"
    python ingest.py --include-hidden
"""
import argparse
from collections import Counter

import osxphotos
import pandas as pd
from tqdm import tqdm

from common import DATA, INDEX

UNKNOWN = "_UNKNOWN_"  # osxphotos name for faces you haven't named


def pick_image(p) -> tuple[str | None, str]:
    """Best local image to index. Previews are smaller and faster than originals."""
    derivs = p.path_derivatives or []
    if derivs:
        return derivs[0], "preview"  # largest preview first
    if p.path and not p.ismissing and not p.ismovie:
        return p.path, "original"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", help="Path to a .photoslibrary (default: system library)")
    ap.add_argument("--include-hidden", action="store_true")
    args = ap.parse_args()

    print("Opening Photos library (first time can take a minute)...")
    db = osxphotos.PhotosDB(args.library) if args.library else osxphotos.PhotosDB()

    rows, sources = [], Counter()
    for p in tqdm(db.photos(), desc="Reading"):
        if p.hidden and not args.include_hidden:
            continue
        img, src = pick_image(p)
        sources[src] += 1
        place = p.place
        rows.append(
            {
                "uuid": p.uuid,
                "date": p.date,
                "date_local": p.date.strftime("%Y-%m-%d"),  # local day, for holiday windows
                "is_video": p.ismovie,
                "persons": [n for n in p.persons if n and n != UNKNOWN],
                "n_faces": len(p.persons),
                "labels": list(p.labels),        # Apple's own scene labels (baseline)
                "albums": list(p.albums),
                "keywords": list(p.keywords),
                "title": p.title,
                "place": place.name if place else None,
                "lat": p.latitude,
                "lon": p.longitude,
                "image_path": img,
                "image_source": src,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    DATA.mkdir(exist_ok=True)
    df.to_parquet(INDEX, index=False)

    named = Counter(n for ps in df["persons"] for n in ps)
    print(f"\nSaved {len(df):,} items to {INDEX}")
    print(f"  photos: {(~df.is_video).sum():,}  videos: {df.is_video.sum():,}")
    print(f"  image source: {dict(sources)}")
    print(f"  named people: {len(named)}  (top: {', '.join(n for n, _ in named.most_common(8))})")
    if sources["none"]:
        print(f"  {sources['none']:,} items have no local image yet and will be skipped.")


if __name__ == "__main__":
    main()
