"""Interpretability: read the learned linear model as linguistic evidence.

Because the perceptron is linear and its inputs are *named, standardized*
features, a feature's weight is a direct, unit-comparable measure of how strongly
that cue drives a headline toward clickbait (positive weight) or toward
legitimate news (negative weight). This module ranks and visualizes those
weights and produces per-headline explanations — the concrete realization of the
project's goal of "exposing which linguistic cues most strongly signal
clickbait".

Standardization matters here: weights are only comparable across features if the
features share a scale, which is why the pipeline standardizes inputs before
training and this module reports weights in that standardized space.
"""

from __future__ import annotations

import numpy as np

from . import plotting_style as ps


def rank_weights(w: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return ``(feature, weight)`` pairs sorted from most clickbait-positive
    to most clickbait-negative."""
    pairs = list(zip(feature_names, [float(v) for v in w]))
    return sorted(pairs, key=lambda kv: kv[1], reverse=True)


def plot_feature_weights(
    w: np.ndarray, feature_names: list[str], path: str, title: str | None = None
) -> None:
    """Diverging horizontal bar chart of all feature weights, sorted.

    Red bars = cues that push toward *clickbait*; blue bars = cues that push
    toward *legitimate news*. Longer bar = stronger influence.
    """
    import matplotlib.pyplot as plt

    ps.apply_style()
    ranked = rank_weights(w, feature_names)
    names = [n for n, _ in ranked]
    values = np.array([v for _, v in ranked])

    # Plot ascending so the strongest clickbait cue sits at the top.
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
    ax.set_xlabel("Standardized weight  (←  legitimate      clickbait  →)")
    ax.set_title(title or "Which cues signal clickbait? (perceptron weights)")

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
    """Grouped bars: mean raw feature value per class, for the most separating
    features (ranked by absolute mean difference)."""
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mean_cb = X[y == 1].mean(axis=0)
    mean_no = X[y == 0].mean(axis=0)
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
    ax.set_xlabel("Mean feature value")
    ax.set_title(f"Most discriminative cues — class means (top {top_k})")
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

    Contribution of feature *j* = standardized_value_j * weight_j. Summed with
    the bias, these give the decision score, so the largest-magnitude
    contributions explain *why* this headline was classified as it was.
    """
    x_raw = extractor.transform([headline])
    x_std = scaler.transform(x_raw)[0]
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
        "top_contributions": top,
    }
