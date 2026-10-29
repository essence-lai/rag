import hashlib
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised float32 row per text."""
        ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Deterministic hashing-trick bag of words. No model download; for tests."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                out[row, value % self.dim] += 1.0 if value >> 63 else -1.0
        return normalize(out)


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed. The model loads on first use."""

    def __init__(self, model: str = DEFAULT_MODEL, dim: int = 384) -> None:
        self.name = f"fastembed:{model}"
        self.dim = dim
        self._model_name = model
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.array(list(self._model.embed(texts, batch_size=64))))
