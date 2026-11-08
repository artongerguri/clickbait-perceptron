"""Paired model comparisons and controlled capitalization stress tests.

Cross-validation receives only the outer training partition. Every estimator
uses the same folds, and every learned transformation is fitted inside its
training fold. Capitalization experiments instead keep the outer test set and
its labels fixed, changing only casing after training on the original texts.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import plotting_style as ps
from .baselines import TfidfLogisticRegression
from .evaluate import compute_metrics
from .features import FeatureExtractor
from .perceptron import AveragedPerceptron
from .preprocessing import _WORD_RE, clean


_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
_CAPITALIZATION_FEATURES = {
    "title_case_ratio", "all_caps_word_count", "all_caps_ratio",
}


def _checked_corpus(headlines, labels):
    texts = list(headlines)
    y = np.asarray(labels)
    if y.ndim != 1 or y.size != len(texts):
        raise ValueError("Headlines and 1D labels must have matching lengths.")
    if not texts:
        raise ValueError("Experiments require at least one headline.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Experiments require both binary labels 0 and 1.")
    return np.asarray(texts, dtype=object), y


def compare_cross_validation(
    headlines, labels, *, epochs: int = 30, folds: int = 5, seed: int = 42,
) -> dict:
    """Compare three models on paired folds of the supplied training partition.

    The engineered-feature logistic regression controls for the classifier;
    TF-IDF logistic regression compares a different representation. Returned
    perceptron weights are in each fold's standardized feature coordinates.
    """
    texts, y = _checked_corpus(headlines, labels)
    if isinstance(folds, bool) or not isinstance(folds, Integral) or folds < 2:
        raise ValueError("folds must be an integer of at least 2.")
    if min(np.bincount(y.astype(int))) < folds:
        raise ValueError("Each class must contain at least folds samples.")

    features = FeatureExtractor().transform(texts)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {
        name: {metric: [] for metric in _METRICS}
        for name in (
            "averaged_perceptron", "engineered_logreg", "tfidf_logreg_baseline",
        )
    }
    fold_weights = []

    for train_idx, validation_idx in splitter.split(texts, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_validation = scaler.transform(features[validation_idx])
        perceptron = AveragedPerceptron(max_epochs=epochs, random_state=seed)
        engineered_logreg = LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=seed,
        )
        tfidf_logreg = TfidfLogisticRegression(random_state=seed)
        models = (
            ("averaged_perceptron", perceptron, X_train, X_validation),
            ("engineered_logreg", engineered_logreg, X_train, X_validation),
            (
                "tfidf_logreg_baseline", tfidf_logreg,
                texts[train_idx].tolist(), texts[validation_idx].tolist(),
            ),
        )
        for name, model, train_input, validation_input in models:
            model.fit(train_input, y[train_idx])
            metrics = compute_metrics(
                y[validation_idx], model.predict(validation_input),
                model.decision_function(validation_input),
            )
            for metric in _METRICS:
                scores[name][metric].append(metrics[metric])
        fold_weights.append(perceptron.w_.tolist())

    return {
        "n_samples": len(texts),
        "n_folds": int(folds),
        "models": {
            name: {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "fold_scores": values,
                }
                for metric, values in model_scores.items()
            }
            for name, model_scores in scores.items()
        },
        "perceptron_fold_weights": fold_weights,
    }


def _capitalized_tokens(headline: str) -> str:
    """Uppercase only token initials; preserve the rest of each token.

    Using the project's tokenizer pattern keeps contractions and hyphenated
    words intact. Cleaning matches the feature extractor's normalization.
    """
    return _WORD_RE.sub(
        lambda match: match.group(0)[:1].upper() + match.group(0)[1:],
        clean(headline),
    )


def run_capitalization_experiments(
    train_headlines, train_labels, test_headlines, test_labels,
    *, epochs: int = 30, seed: int = 42,
) -> dict:
    """Train three feature variants, then stress-test their frozen predictions.

    ``prediction_changes_vs_original`` counts changed predictions on exactly
    the same test examples. Sentence case uses ``str.capitalize()`` (and hence
    lowercases remaining letters); capitalized tokens retain existing acronyms.
    These controlled perturbations measure casing sensitivity, not performance
    on a separately collected source or new labels.
    """
    train_texts, y_train = _checked_corpus(train_headlines, train_labels)
    test_texts, y_test = _checked_corpus(test_headlines, test_labels)
    extractor = FeatureExtractor()
    feature_names = extractor.feature_names
    X_train = extractor.transform(train_texts)
    conditions = {
        "original": test_texts,
        "sentence_case": [text.capitalize() for text in test_texts],
        "capitalized_tokens": [_capitalized_tokens(text) for text in test_texts],
    }
    test_features = {
        condition: extractor.transform(texts)
        for condition, texts in conditions.items()
    }
    variants = {
        "full": list(range(len(feature_names))),
        "without_capitalization": [
            i for i, name in enumerate(feature_names)
            if name not in _CAPITALIZATION_FEATURES
        ],
        "title_case_only": [feature_names.index("title_case_ratio")],
    }
    results = {}
    for variant, columns in variants.items():
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(X_train[:, columns])
        model = AveragedPerceptron(max_epochs=epochs, random_state=seed)
        model.fit(scaled_train, y_train)
        condition_metrics = {}
        original_predictions = None
        for condition, matrix in test_features.items():
            scaled_test = scaler.transform(matrix[:, columns])
            predictions = model.predict(scaled_test)
            if original_predictions is None:
                original_predictions = predictions.copy()
            metrics = compute_metrics(
                y_test, predictions, model.decision_function(scaled_test),
            )
            metrics["prediction_changes_vs_original"] = int(
                np.count_nonzero(predictions != original_predictions)
            )
            condition_metrics[condition] = metrics
        results[variant] = {
            "n_features": len(columns),
            "feature_names": [feature_names[i] for i in columns],
            "conditions": condition_metrics,
        }
    return {
        "variants": results,
        "protocol": (
            "Each feature variant is trained on original training headlines only. "
            "Its model and training-only scaler are frozen across three versions "
            "of the same held-out headlines, with the same labels: original, "
            "sentence case (str.capitalize), and capitalized token initials "
            "(remaining characters preserved). prediction_changes_vs_original "
            "is a count of changed predictions. This is a controlled casing "
            "sensitivity test, not an evaluation on new publishers."
        ),
    }


def plot_capitalization_robustness(experiments: dict, path: str) -> None:
    """Plot held-out accuracy for all feature variants and casing conditions."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    variants = ["full", "without_capitalization", "title_case_only"]
    conditions = ["original", "sentence_case", "capitalized_tokens"]
    condition_labels = ["Original", "Sentence case", "Capitalized tokens"]
    x = np.arange(len(variants))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for offset, (condition, label) in enumerate(zip(conditions, condition_labels)):
        accuracies = [
            experiments["variants"][variant]["conditions"][condition]["accuracy"] * 100
            for variant in variants
        ]
        bars = ax.bar(
            x + (offset - 1) * width, accuracies, width=width,
            color=ps.CATEGORICAL[offset], label=label,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_xticks(x, ["All features", "Without capitalization", "Title-case ratio only"])
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Capitalization sensitivity with frozen models")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    ps.save(fig, path)
