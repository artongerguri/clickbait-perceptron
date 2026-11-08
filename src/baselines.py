"""High-dimensional linear reference model: TF-IDF + Logistic Regression.

TF-IDF over word 1–2 grams provides a standard text-classification baseline.
Its individual word/ngram coefficients can be inspected, though a vocabulary
with thousands of features is harder to summarize than 32 named linguistic
cues. This comparison therefore contrasts feature representations as well as
classifiers; a logistic-regression control on the same engineered features
helps separate their contributions.

We expose the same scikit-learn-style API as :class:`~src.perceptron.AveragedPerceptron`
so both models can be dropped into the shared evaluation code unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogisticRegression:
    """Inspectable, high-dimensional TF-IDF + Logistic Regression baseline."""

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=ngram_range,
                        max_features=max_features,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=1000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, headlines, y) -> "TfidfLogisticRegression":
        self.pipeline.fit(headlines, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict(self, headlines) -> np.ndarray:
        return self.pipeline.predict(headlines)

    def decision_function(self, headlines) -> np.ndarray:
        return self.pipeline.decision_function(headlines)

    def score(self, headlines, y) -> float:
        return float(self.pipeline.score(headlines, y))

    @property
    def n_features(self) -> int:
        """Size of the learned vocabulary (the TF-IDF feature space)."""
        return len(self.pipeline.named_steps["tfidf"].vocabulary_)
