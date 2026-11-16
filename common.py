"""Shared paths and model loading for Alaala."""
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DATA = ROOT / "data"
INDEX = DATA / "photos.parquet"          # one row per photo/video (from ingest.py)
EMB = DATA / "embeddings.npy"            # float16 [n, d], row-aligned with EMB_IDS
EMB_IDS = DATA / "embedding_uuids.npy"   # uuid per embedding row
CHUNKS = DATA / "emb_chunks"             # resumable partial results

# Small + fast, fits on an 8GB M2. Swap to "MobileCLIP2-S0" if memory is tight,
# or "MobileCLIP2-B" if you want more accuracy and can spare the RAM.
MODEL_NAME = "MobileCLIP2-S2"
PRETRAINED = "dfndr2b"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Return (model, preprocess, tokenizer, device). Downloads weights on first run."""
    import open_clip
    from timm.utils import reparameterize_model

    dev = device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()
    model = reparameterize_model(model)  # faster inference, same outputs
    model = model.to(dev)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, dev
