"""An averaged Perceptron classifier implemented from scratch with NumPy.

The classic Perceptron (Rosenblatt, 1958) is an online, mistake-driven linear
classifier. We implement the *averaged* variant (Freund & Schapire, 1999),
which returns the average of all intermediate weight vectors instead of the last
one. Averaging dramatically reduces variance and typically yields far better
generalization at essentially no extra inference cost, while remaining a *linear*
and therefore fully *interpretable* model: the learned weight of each engineered
feature is a direct, signed measure of how strongly that cue pushes a headline
toward "clickbait" (positive) or "not clickbait" (negative).

The efficient averaging trick (a single cached accumulator rather than summing
every intermediate weight vector) follows Hal Daumé III's formulation.

The public API deliberately mirrors scikit-learn (`fit`/`predict`/`score`) so the
model is a drop-in component in the evaluation pipeline and comparable to the
black-box baseline on equal footing.
"""

from __future__ import annotations

from numbers import Integral, Number, Real

import numpy as np


class AveragedPerceptron:
    """Binary averaged Perceptron.

    Parameters
    ----------
    max_epochs:
        Maximum number of passes over the training data.
    learning_rate:
        Positive scale applied to each weight update. From zero initialization,
        this rescales the parameters without changing the decision boundary
        (up to floating-point effects).
    shuffle:
        Whether to reshuffle the data each epoch (recommended for online
        learning to avoid ordering bias).
    averaged:
        If ``True`` use the averaged weights for prediction; otherwise use the
        last weights (the vanilla Perceptron). Exposed so the report can compare
        the two variants.
    random_state:
        Seed for reproducible shuffling.
    verbose:
        Print per-epoch update counts and training accuracy.

    Attributes
    ----------
    w_ , b_:
        Weight vector and bias used for prediction (averaged or last).
    n_updates_:
        Total number of mistakes (weight updates) made during training.
    history_:
        Dict with ``epoch_updates`` and ``epoch_train_accuracy`` lists, used for
        convergence plots.
    """

    def __init__(
        self,
        max_epochs: int = 30,
        learning_rate: float = 1.0,
        shuffle: bool = True,
        averaged: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.shuffle = shuffle
        self.averaged = averaged
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AveragedPerceptron":
        """Fit on a finite 2D feature matrix and two distinct class labels."""
        if (
            isinstance(self.max_epochs, bool)
            or not isinstance(self.max_epochs, Integral)
            or self.max_epochs <= 0
        ):
            raise ValueError("max_epochs must be a positive integer.")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a finite positive number.")

        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("AveragedPerceptron supports binary classification only.")

        # Map the two class labels to {-1, +1} for the margin formulation.
        self.classes_ = classes
        self._label_pos = self.classes_[1]
        self._label_neg = self.classes_[0]
        y_pm = np.where(y == self._label_pos, 1.0, -1.0)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Current and cached (for averaging) parameters.
        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0
        u = np.zeros(n_features, dtype=np.float64)  # cached weighted sum
        beta = 0.0
        counter = 1.0
        lr = self.learning_rate

        self.history_ = {"epoch_updates": [], "epoch_train_accuracy": []}
        self.n_updates_ = 0

        for epoch in range(self.max_epochs):
            order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            updates = 0

            for i in order:
                xi = X[i]
                yi = y_pm[i]
                # Mistake if the (signed) margin is non-positive.
                if yi * (w @ xi + b) <= 0.0:
                    w += lr * yi * xi
                    b += lr * yi
                    u += lr * yi * counter * xi
                    beta += lr * yi * counter
                    updates += 1
                counter += 1.0

            self.n_updates_ += updates

            # Weights currently in effect for this epoch's bookkeeping.
            if self.averaged:
                w_eff = w - u / counter
                b_eff = b - beta / counter
            else:
                w_eff, b_eff = w, b

            train_acc = self._accuracy_from_params(X, y_pm, w_eff, b_eff)
            self.history_["epoch_updates"].append(int(updates))
            self.history_["epoch_train_accuracy"].append(float(train_acc))

            if self.verbose:
                print(
                    f"  epoch {epoch + 1:>3}/{self.max_epochs}: "
                    f"updates={updates:>6d}  train_acc={train_acc:.4f}"
                )

            # Convergence: a full pass with no mistakes => data linearly separated.
            if updates == 0:
                if self.verbose:
                    print(f"  converged after {epoch + 1} epochs (0 updates).")
                break

        # Freeze the parameters used for prediction.
        if self.averaged:
            self.w_ = w - u / counter
            self.b_ = b - beta / counter
        else:
            self.w_ = w.copy()
            self.b_ = b
        self.w_last_, self.b_last_ = w.copy(), b
        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance-like score; ``> 0`` means the positive class."""
        if not hasattr(self, "w_"):
            raise ValueError("The model is not fitted. Call fit before prediction.")
        X = self._validate_X(X, allow_empty=True)
        if X.shape[1] != self.w_.size:
            raise ValueError(f"X must have {self.w_.size} features; got {X.shape[1]}.")
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the original class labels."""
        scores = self.decision_function(X)
        return self.classes_[(scores > 0.0).astype(np.intp)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy on ``(X, y)``."""
        predictions = self.predict(X)
        if predictions.size == 0:
            raise ValueError("Scoring requires at least one sample.")
        labels = self._validate_y(y, predictions.size)
        return float(np.mean(predictions == labels))

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_X(X: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not allow_empty and X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1D label array or a single column.")
        if y.size != n_samples:
            raise ValueError("X and y must contain the same number of samples.")
        if any(label is None for label in y):
            raise ValueError("y must not contain missing labels.")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("y must contain only finite labels.")
        if y.dtype == object and any(
            isinstance(label, Number) and (label != label or abs(label) == np.inf)
            for label in y
        ):
            raise ValueError("y must contain only finite labels.")
        return y

    @staticmethod
    def _accuracy_from_params(
        X: np.ndarray, y_pm: np.ndarray, w: np.ndarray, b: float
    ) -> float:
        preds = np.where(X @ w + b > 0.0, 1.0, -1.0)
        return float(np.mean(preds == y_pm))
