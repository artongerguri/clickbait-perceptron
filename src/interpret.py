"""Inspect standardized model coefficients and decompose predictions.

A coefficient gives the score change for a one-standard-deviation increase in
one feature while holding the others fixed. Standardization puts features on a
shared scale, but correlated cues can redistribute coefficient magnitudes and
signs across fits. These are conditional descriptions of a fitted model, not
causal effects or unique measures of linguistic importance. Fold summaries help
show how stable those descriptions are under different training samples.
"""

from __future__ import annotations

import numpy as np

from . import plotting_style as ps


def rank_weights(w: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return ``(feature, weight)`` pairs sorted from most clickbait-positive
    to most clickbait-negative."""
    pairs = list(zip(feature_names, [float(v) for v in w]))
    return sorted(pairs, key=lambda kv: kv[1], reverse=True)


def summarize_weight_stability(
    fold_weights: np.ndarray,
    feature_names: list[str],
    reference_weights: np.ndarray,
) -> list[dict]:
    """Summarize standardized coefficients across independently trained folds.

    The population standard deviation describes variation across the supplied
    folds. Sign agreement is the fraction matching the full-training reference
    coefficient's sign; a zero coefficient is treated as a separate sign.
    Feature order must be identical in every fold and in the reference fit.
    """
    weights = np.asarray(fold_weights, dtype=float)
    reference = np.asarray(reference_weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] == 0:
        raise ValueError("fold_weights must be a nonempty 2D matrix.")
    if reference.ndim != 1 or weights.shape[1] != reference.size:
        raise ValueError("Fold and reference weights must have the same feature width.")
    if len(feature_names) != reference.size or reference.size == 0:
        raise ValueError("feature_names must match the nonempty weight vectors.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique.")
    if not np.isfinite(weights).all() or not np.isfinite(reference).all():
        raise ValueError("Weights must contain only finite values.")
    means = weights.mean(axis=0)
    stds = weights.std(axis=0)
    agreements = (np.sign(weights) == np.sign(reference)).mean(axis=0)
    return [
        {
            "feature": name,
            "mean_weight": float(means[j]),
            "std_weight": float(stds[j]),
            "sign_agreement": float(agreements[j]),
        }
        for j, name in enumerate(feature_names)
    ]


def plot_feature_weights(
    w: np.ndarray, feature_names: list[str], path: str, title: str | None = None
) -> None:
    """Diverging horizontal bar chart of all feature weights, sorted.

    Red bars = positive coefficients; blue bars = negative coefficients.
    Magnitudes describe conditional score changes on standardized features.
    """
    import matplotlib.pyplot as plt

    ps.apply_style()
    ranked = rank_weights(w, feature_names)
    names = [n for n, _ in ranked]
    values = np.array([v for _, v in ranked])

    # Plot ascending so the most positive coefficient sits at the top.
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = values[order]

    colors = [ps.DIVERGING_POS if v > 0 else ps.DIVERGING_NEG for v in values]

    fig_h = max(5.0, 0.32 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, height=0.68, zorder=3)
    ax.axvline(0, color=ps.BASELINE, linewidth=1.0, zorder=2)

    ax.set_yticks(y_pos, names, fontsize=9)
    ax.set_xlabel("Standardized coefficient  (←  not clickbait      clickbait  →)")
    ax.set_title(title or "Model coefficients (averaged perceptron)")

    xmax = np.abs(values).max()
    for yi, v in zip(y_pos, values):
        offset = 0.01 * xmax if v >= 0 else -0.01 * xmax
        ax.text(
            v + offset, yi, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right",
            fontsize=8, color=ps.INK_SECONDARY,
        )
    ax.set_xlim(-1.25 * xmax, 1.25 * xmax)
    ps.save(fig, path)


def plot_class_feature_means(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    path: str,
    top_k: int = 12,
) -> None:
    """Compare standardized class means, ranked by absolute mean difference.

    The supplied training matrix defines each feature's center and scale;
    constant columns use scale one. This keeps selection and bar widths
    independent of the raw units of the engineered features.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    X_std = (X - X.mean(axis=0)) / scale
    mean_cb = X_std[y == 1].mean(axis=0)
    mean_no = X_std[y == 0].mean(axis=0)
    diff = np.abs(mean_cb - mean_no)
    top = np.argsort(diff)[::-1][:top_k]
    # Order the selected features by the clickbait mean for readability.
    top = top[np.argsort(mean_cb[top])]

    names = [feature_names[i] for i in top]
    ps.apply_style()
    fig_h = max(4.0, 0.42 * len(names) + 1.0)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    y_pos = np.arange(len(names))
    h = 0.38
    ax.barh(y_pos + h / 2, mean_cb[top], height=h, color=ps.DIVERGING_POS,
            label="Clickbait", zorder=3)
    ax.barh(y_pos - h / 2, mean_no[top], height=h, color=ps.DIVERGING_NEG,
            label="Not clickbait", zorder=3)
    ax.set_yticks(y_pos, names, fontsize=9)
    ax.set_xlabel("Mean standardized feature value")
    ax.set_title(f"Largest standardized class differences (top {top_k})")
    ax.legend(loc="lower right", fontsize=9)
    ps.save(fig, path)


def explain_headline(
    extractor,
    scaler,
    w: np.ndarray,
    b: float,
    feature_names: list[str],
    headline: str,
    top_k: int = 6,
) -> dict:
    """Decompose a single prediction into per-feature contributions.

    Contribution of feature *j* = standardized_value_j * weight_j. The displayed
    contributions plus ``other_contribution`` and ``bias`` reconstruct the
    decision score, even when ``top_k`` omits features. This is a decomposition
    of this model's score rather than a causal explanation of the headline.
    """
    x_raw = extractor.transform([headline])
    x_std = scaler.transform(x_raw)[0]
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or w.size != x_std.size or len(feature_names) != w.size:
        raise ValueError("Weights, feature names and transformed values must align.")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, (int, np.integer))
        or top_k < 0
    ):
        raise ValueError("top_k must be a nonnegative integer.")
    contrib = x_std * w
    score = float(contrib.sum() + b)

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    top = [
        {
            "feature": feature_names[j],
            "raw_value": float(x_raw[0, j]),
            "contribution": float(contrib[j]),
        }
        for j in order
    ]
    return {
        "headline": headline,
        "decision_score": score,
        "prediction": "clickbait" if score > 0 else "not clickbait",
        "bias": float(b),
        "other_contribution": float(contrib.sum() - contrib[order].sum()),
        "top_contributions": top,
    }
