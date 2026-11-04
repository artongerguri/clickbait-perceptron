"""Evaluation: metrics, cross-validation, and figures.

We report the standard battery for a binary classifier — accuracy, precision,
recall, F1 and ROC-AUC — plus a confusion matrix, rather than a single headline
number, because on a balanced corpus precision/recall still reveal *which* kind
of error each model makes. Stratified k-fold cross-validation quantifies the
stability of the engineered-feature perceptron.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import plotting_style as ps


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Return a dict of scalar metrics (+ confusion matrix counts)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def cross_validate_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    k: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified k-fold CV for a feature-matrix model.

    ``model_factory`` is a zero-arg callable returning a fresh estimator so each
    fold trains from scratch. Standardization is fit on the training fold only,
    preventing leakage.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    per_fold = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        model = model_factory()
        model.fit(X_tr, y[train_idx])
        y_pred = model.predict(X_te)
        y_score = model.decision_function(X_te)

        m = compute_metrics(y[test_idx], y_pred, y_score)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            per_fold[key].append(m[key])

    return {
        key: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for key, vals in per_fold.items()
    }


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm_counts: dict, path: str, title: str) -> None:
    """Annotated confusion-matrix heatmap (sequential blue)."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    cm = np.array(
        [[cm_counts["tn"], cm_counts["fp"]], [cm_counts["fn"], cm_counts["tp"]]],
        dtype=float,
    )
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.grid(False)
    im = ax.imshow(cm_norm, cmap=ps.sequential_cmap(), vmin=0, vmax=1)

    classes = ["Not clickbait", "Clickbait"]
    ax.set_xticks([0, 1], classes)
    ax.set_yticks([0, 1], classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            frac = cm_norm[i, j]
            txt_color = "#ffffff" if frac > 0.5 else ps.INK_PRIMARY
            ax.text(
                j, i,
                f"{int(cm[i, j]):,}\n({frac:.1%})",
                ha="center", va="center",
                color=txt_color, fontsize=11, fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=ps.INK_MUTED, labelcolor=ps.INK_MUTED)
    ps.save(fig, path)


def plot_roc_curves(curves: list[tuple[str, np.ndarray, np.ndarray, float]], path: str) -> None:
    """Overlay ROC curves for several models (categorical colors)."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))

    for idx, (name, fpr, tpr, auc) in enumerate(curves):
        ax.plot(
            fpr, tpr,
            color=ps.CATEGORICAL[idx % len(ps.CATEGORICAL)],
            linewidth=2.0,
            label=f"{name} (AUC = {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color=ps.INK_MUTED, linewidth=1.0, linestyle="--",
            label="Chance (AUC = 0.500)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    ps.save(fig, path)


def roc_points(y_true, y_score) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) for an ROC overlay."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(roc_auc_score(y_true, y_score))


def plot_convergence(history: dict, path: str) -> None:
    """Two-panel convergence plot: training accuracy and mistakes per epoch."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    epochs = np.arange(1, len(history["epoch_train_accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax1.plot(epochs, history["epoch_train_accuracy"], color=ps.BLUE,
             linewidth=2.0, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training accuracy")
    ax1.set_title("Convergence — accuracy")

    ax2.plot(epochs, history["epoch_updates"], color=ps.ORANGE,
             linewidth=2.0, marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weight updates (mistakes)")
    ax2.set_title("Convergence — mistakes")

    fig.tight_layout()
    ps.save(fig, path)
