"""Checks for paired, leakage-free model and capitalization comparisons."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src import experiments
from src.baselines import TfidfLogisticRegression
from src.features import FeatureExtractor
from src.perceptron import AveragedPerceptron
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def corpus():
    texts = [
        "amber Senate approves budget after lengthy negotiations",
        "bravo You Won't Believe These Amazing Things!",
        "cedar Market falls sharply",
        "delta 17 Tricks That Will Make You Laugh",
        "elm Scientists discover new species of frog near river",
        "foxtrot This Is Why You Need To Know Everything",
        "garnet Minister announces policy",
        "hazel Which Movie Character Are You?",
        "indigo Council opens new library downtown today",
        "juniper The Best Photos You Will Ever See!",
        "kelp NASA releases mission report after review",
        "lilac OMG These Adorable Dogs Will Melt Your Heart",
    ]
    return texts, np.array([0, 1] * 6)


def test_cross_validation_pairs_models_and_keeps_preprocessing_inside_folds(
    monkeypatch, corpus,
):
    texts, labels = corpus
    perceptron_records, logreg_records, tfidf_records, scaler_records = [], [], [], []

    class ObservedScaler(StandardScaler):
        def fit(self, X, y=None, **kwargs):
            result = super().fit(X, y, **kwargs)
            scaler_records.append((np.array(X), self.mean_.copy(), self.scale_.copy()))
            return result

    class ObservedPerceptron(AveragedPerceptron):
        def fit(self, X, y):
            self.record = {"train": np.array(X), "labels": np.array(y)}
            perceptron_records.append(self.record)
            return super().fit(X, y)

        def predict(self, X):
            self.record["validation"] = np.array(X)
            return super().predict(X)

    class ObservedLogreg(LogisticRegression):
        def fit(self, X, y, **kwargs):
            self.record = {"train": np.array(X), "labels": np.array(y)}
            logreg_records.append(self.record)
            return super().fit(X, y, **kwargs)

        def predict(self, X):
            self.record["validation"] = np.array(X)
            return super().predict(X)

    class ObservedTfidf(TfidfLogisticRegression):
        def fit(self, headlines, y):
            self.record = {"train": list(headlines), "labels": np.array(y)}
            tfidf_records.append(self.record)
            result = super().fit(headlines, y)
            self.record["vocabulary"] = set(self.pipeline.named_steps["tfidf"].vocabulary_)
            return result

        def predict(self, headlines):
            self.record["validation"] = list(headlines)
            return super().predict(headlines)

    monkeypatch.setattr(experiments, "StandardScaler", ObservedScaler)
    monkeypatch.setattr(experiments, "AveragedPerceptron", ObservedPerceptron)
    monkeypatch.setattr(experiments, "LogisticRegression", ObservedLogreg)
    monkeypatch.setattr(experiments, "TfidfLogisticRegression", ObservedTfidf)
    result = experiments.compare_cross_validation(texts, labels, epochs=3, folds=3)

    assert result["n_samples"] == len(texts)
    assert result["n_folds"] == 3
    assert len(perceptron_records) == len(logreg_records) == len(tfidf_records) == 3
    assert len(scaler_records) == 3
    extractor = FeatureExtractor()
    validation_occurrences = []
    for perceptron, logistic, tfidf, (raw_train, mean, scale) in zip(
        perceptron_records, logreg_records, tfidf_records, scaler_records,
    ):
        # The two numeric models see identical paired training and validation data.
        np.testing.assert_array_equal(perceptron["train"], logistic["train"])
        np.testing.assert_array_equal(perceptron["validation"], logistic["validation"])
        np.testing.assert_array_equal(perceptron["labels"], logistic["labels"])
        np.testing.assert_array_equal(perceptron["labels"], tfidf["labels"])
        assert not set(tfidf["train"]) & set(tfidf["validation"])
        assert set(tfidf["train"]) | set(tfidf["validation"]) == set(texts)
        validation_occurrences.extend(tfidf["validation"])
        # A unique word in each headline reveals any vocabulary fit on held-out text.
        for headline in tfidf["validation"]:
            assert headline.split()[0] not in tfidf["vocabulary"]
        for headline in tfidf["train"]:
            assert headline.split()[0] in tfidf["vocabulary"]
        np.testing.assert_allclose(raw_train, extractor.transform(tfidf["train"]))
        np.testing.assert_allclose(mean, raw_train.mean(axis=0))
        expected_validation = (extractor.transform(tfidf["validation"]) - mean) / scale
        np.testing.assert_allclose(perceptron["validation"], expected_validation)
    assert sorted(validation_occurrences) == sorted(texts)
    global_mean = extractor.transform(texts).mean(axis=0)
    assert any(not np.allclose(mean, global_mean) for _, mean, _ in scaler_records)
    assert np.asarray(result["perceptron_fold_weights"]).shape == (3, extractor.n_features)
    for model in result["models"].values():
        assert set(model) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for metric in model.values():
            assert len(metric["fold_scores"]) == 3
            assert metric["mean"] == pytest.approx(np.mean(metric["fold_scores"]))
            assert metric["std"] == pytest.approx(np.std(metric["fold_scores"]))
    json.dumps(result)  # The pipeline writes the results to JSON.


def test_cross_validation_is_reproducible(corpus):
    texts, labels = corpus
    first = experiments.compare_cross_validation(texts, labels, epochs=3, folds=3)
    second = experiments.compare_cross_validation(texts, labels, epochs=3, folds=3)
    assert first == second


@pytest.mark.parametrize("folds", [1, 0, 2.5, True, 7])
def test_cross_validation_rejects_invalid_or_underpopulated_folds(corpus, folds):
    with pytest.raises(ValueError):
        experiments.compare_cross_validation(*corpus, epochs=1, folds=folds)


def test_capitalization_preserves_labels_and_fits_once_per_feature_variant(
    monkeypatch, corpus,
):
    texts, labels = corpus
    train_texts, train_labels = texts[:8], labels[:8]
    test_texts, test_labels = texts[8:], labels[8:]
    fitted_models, fitted_scalers, metric_labels = [], [], []
    original_compute_metrics = experiments.compute_metrics

    class ObservedPerceptron(AveragedPerceptron):
        def fit(self, X, y):
            self.test_inputs = []
            fitted_models.append((np.array(X), np.array(y), self.test_inputs))
            return super().fit(X, y)

        def predict(self, X):
            self.test_inputs.append(np.array(X))
            return super().predict(X)

    class ObservedScaler(StandardScaler):
        def fit(self, X, y=None, **kwargs):
            fitted_scalers.append(np.array(X))
            return super().fit(X, y, **kwargs)

    def record_metrics(y_true, *args, **kwargs):
        metric_labels.append(np.array(y_true))
        return original_compute_metrics(y_true, *args, **kwargs)

    monkeypatch.setattr(experiments, "AveragedPerceptron", ObservedPerceptron)
    monkeypatch.setattr(experiments, "StandardScaler", ObservedScaler)
    monkeypatch.setattr(experiments, "compute_metrics", record_metrics)
    result = experiments.run_capitalization_experiments(
        train_texts, train_labels, test_texts, test_labels, epochs=3,
    )
    assert len(fitted_models) == len(fitted_scalers) == 3
    assert len(metric_labels) == 9
    for y_true in metric_labels:
        np.testing.assert_array_equal(y_true, test_labels)
    extractor = FeatureExtractor()
    cap_names = {"title_case_ratio", "all_caps_word_count", "all_caps_ratio"}
    expected_names = {
        "full": extractor.feature_names,
        "without_capitalization": [n for n in extractor.feature_names if n not in cap_names],
        "title_case_only": ["title_case_ratio"],
    }
    capitalized_test_texts = [
        "Indigo Council Opens New Library Downtown Today",
        "Juniper The Best Photos You Will Ever See!",
        "Kelp NASA Releases Mission Report After Review",
        "Lilac OMG These Adorable Dogs Will Melt Your Heart",
    ]
    for (name, names), raw_train, (scaled_train, fitted_labels, prediction_inputs) in zip(
        expected_names.items(), fitted_scalers, fitted_models,
    ):
        variant = result["variants"][name]
        assert variant["feature_names"] == names
        assert variant["n_features"] == len(names)
        columns = [extractor.feature_names.index(n) for n in names]
        np.testing.assert_allclose(raw_train, extractor.transform(train_texts)[:, columns])
        assert raw_train.shape == scaled_train.shape == (len(train_texts), len(names))
        np.testing.assert_array_equal(fitted_labels, train_labels)
        assert len(prediction_inputs) == 3
        training_scaler = StandardScaler().fit(raw_train)
        for observed_input, expected_texts in zip(
            prediction_inputs,
            [test_texts, [text.capitalize() for text in test_texts], capitalized_test_texts],
        ):
            expected_matrix = extractor.transform(expected_texts)[:, columns]
            np.testing.assert_allclose(
                observed_input, training_scaler.transform(expected_matrix),
            )
        assert set(variant["conditions"]) == {"original", "sentence_case", "capitalized_tokens"}
        assert variant["conditions"]["original"]["prediction_changes_vs_original"] == 0
        for metrics in variant["conditions"].values():
            assert 0 <= metrics["prediction_changes_vs_original"] <= len(test_texts)
    # Removing every casing feature must make casing-only perturbations invisible.
    no_caps = result["variants"]["without_capitalization"]["conditions"]
    assert no_caps["original"] == no_caps["sentence_case"] == no_caps["capitalized_tokens"]
    assert "frozen" in result["protocol"]
    json.dumps(result)


def test_capitalized_tokens_preserve_contractions_acronyms_and_hyphens():
    assert experiments._capitalized_tokens("you won't believe NASA's new-world, iPhone!") == (
        "You Won't Believe NASA's New-world, IPhone!"
    )
    assert experiments._capitalized_tokens("you won’t believe this") == "You Won't Believe This"


@pytest.mark.parametrize(
    "texts, labels",
    [([], []), (["one", "two"], [0]), (["one", "two"], [0, 0])],
)
def test_experiments_reject_invalid_corpora(texts, labels):
    with pytest.raises(ValueError):
        experiments.compare_cross_validation(texts, labels, folds=2)
    with pytest.raises(ValueError):
        experiments.run_capitalization_experiments(texts, labels, texts, labels)


def test_capitalization_figure_written(corpus, tmp_path):
    texts, labels = corpus
    result = experiments.run_capitalization_experiments(
        texts[:8], labels[:8], texts[8:], labels[8:], epochs=2,
    )
    out = tmp_path / "capitalization_robustness.png"
    experiments.plot_capitalization_robustness(result, str(out))
    assert out.exists() and out.stat().st_size > 0
