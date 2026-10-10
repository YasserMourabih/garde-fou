"""Frozen multilingual sentence embeddings + logistic regression.

The hypothesis this detector tests: *is the French failure a vocabulary problem
or a semantics problem?* A multilingual encoder maps "ignore the instructions
above" and "ignore les instructions ci-dessus" to nearby vectors even though
they share almost no surface n-grams. If the embedding model transfers to
French while TF-IDF does not, the gap is lexical — and that tells you what to
fix.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from garde_fou.detectors.base import Detector
from garde_fou.textnorm import normalize

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingDetector(Detector):
    name = "embedding"
    kind = "frozen encoder + linear"

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = None
        self.clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [normalize(str(t), aggressive=False) for t in texts]
        return self.encoder.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> EmbeddingDetector:
        self.clf.fit(self.embed(texts), np.asarray(labels))
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty(0, dtype=float)
        return self.clf.predict_proba(self.embed(texts))[:, 1]

    def _save(self, path: Path) -> None:
        with (path / "head.pkl").open("wb") as fh:
            pickle.dump({"clf": self.clf, "model_name": self.model_name}, fh, protocol=5)

    @classmethod
    def load(cls, path: Path) -> EmbeddingDetector:
        with (path / "head.pkl").open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(blob["model_name"])
        obj.clf = blob["clf"]
        return obj
