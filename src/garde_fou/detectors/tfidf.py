"""TF-IDF + logistic regression.

Word n-grams catch the trigger phrases; character n-grams (`char_wb`) are what
make this baseline unexpectedly hard to beat, because `1gn0r3` and `ignore`
share most of their character trigrams once the text is normalised. The
normalisation step is shared with the rule engine so the comparison between the
two is about the classifier, not about preprocessing.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from garde_fou.detectors.base import Detector
from garde_fou.textnorm import normalize


def _build() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     max_features=120_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     sublinear_tf=True, max_features=200_000)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]
    )


class TfidfDetector(Detector):
    name = "tfidf"
    kind = "linear"

    def __init__(self) -> None:
        self.pipeline = _build()

    @staticmethod
    def _prep(texts: Sequence[str]) -> list[str]:
        return [normalize(str(t)) for t in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> TfidfDetector:
        self.pipeline.fit(self._prep(texts), np.asarray(labels))
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty(0, dtype=float)
        return self.pipeline.predict_proba(self._prep(texts))[:, 1]

    def top_features(self, k: int = 15) -> list[tuple[str, float]]:
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        order = np.argsort(coefs)[::-1][:k]
        return [(str(names[i]), float(coefs[i])) for i in order]

    def _save(self, path: Path) -> None:
        with (path / "pipeline.pkl").open("wb") as fh:
            pickle.dump(self.pipeline, fh, protocol=5)

    @classmethod
    def load(cls, path: Path) -> TfidfDetector:
        obj = cls()
        with (path / "pipeline.pkl").open("rb") as fh:
            obj.pipeline = pickle.load(fh)
        return obj
