"""Image and code embeddings for outcome distances (WP4, D-034). Runs on Apple MPS.

Models are read from the local HF cache (``HF_HOME=/Volumes/Models/huggingface``).
Both embedders return float32 arrays of shape (n, d), L2-normalised per row.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

IMAGE_MODEL = "google/siglip2-base-patch16-384"
CODE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CODE_MAX_TOKENS = 16384

os.environ.setdefault("HF_HOME", "/Volumes/Models/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


class ImageEmbedder:
    model_id = IMAGE_MODEL

    def __init__(self, device: str | None = None, batch_size: int = 16):
        from transformers import AutoModel, AutoProcessor

        self.device = _device(device)
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(IMAGE_MODEL)
        self.model = AutoModel.from_pretrained(IMAGE_MODEL).to(self.device).eval()

    def embed(self, paths: list[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        out = []
        for i in range(0, len(paths), self.batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + self.batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                f = self.model.get_image_features(**inputs)
            if not isinstance(f, torch.Tensor):  # newer transformers return an output object
                f = f.pooler_output
            out.append(f.float().cpu().numpy())
        return _normalise(np.concatenate(out)) if out else np.zeros((0, 0), np.float32)


class CodeEmbedder:
    model_id = CODE_MODEL

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(CODE_MODEL, device=_device(device))
        self.model.max_seq_length = CODE_MAX_TOKENS
        self.last_truncated: list[bool] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        tok = self.model.tokenizer
        self.last_truncated = [len(tok(t, add_special_tokens=True)["input_ids"]) > CODE_MAX_TOKENS
                               for t in texts]
        # one text per batch: apps differ a lot in length, padding would waste memory
        E = self.model.encode(texts, batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        return _normalise(E)
