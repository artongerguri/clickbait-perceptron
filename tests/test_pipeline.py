"""Integration tests for evaluation, interpretability and the baseline.

These exercise the pieces that glue the model to the experiment: the metrics
helpers, leak-free cross-validation, the weight-ranking / per-headline
explanation used for interpretability, the TF-IDF + LogReg baseline, and the
figure-writing code paths (smoke-tested so a broken plot fails CI, not the
grader's machine).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # headless, file-only rendering

from src import evaluate, interpret
from src.baselines import TfidfLogisticRegression
from src.features import FeatureExtractor
from src.perceptron import AveragedPerceptron


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_compute_metrics_values(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        y_score = [0.1, 0.6, 0.8, 0.9]
        m = evaluate.compute_metrics(y_true, y_pred, y_score)
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["recall"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(0.8)
        assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_compute_metrics_without_scores_has_no_auc(self):
        m = evaluate.compute_metrics([0, 1], [0, 1])
        assert "roc_auc" not in m


# --------------------------------------------------------------------------- #
# Cross-validation                                                            #
# --------------------------------------------------------------------------- #
class TestCrossValidation:
    def test_cv_structure_and_quality(self):
        rng = np.random.default_rng(0)
        x_pos = rng.normal(2.0, 0.6, size=(60, 2))
        x_neg = rng.normal(-2.0, 0.6, size=(60, 2))
        X = np.vstack([x_pos, x_neg])
        y = np.array([1] * 60 + [0] * 60)

        cv = evaluate.cross_validate_perceptron(
            X, y,
            model_factory=lambda: AveragedPerceptron(max_epochs=20, random_state=0),
            k=3, random_state=0,
        )
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert set(cv[key]) == {"mean", "std"}
        # Clearly separable -> high mean accuracy.
        assert cv["accuracy"]["mean"] > 0.9


# --------------------------------------------------------------------------- #
# Interpretability                                                            #
# --------------------------------------------------------------------------- #
class TestInterpretability:
    def test_rank_weights_sorted_descending(self):
        ranked = interpret.rank_weights(np.array([1.0, -2.0, 0.5]), ["a", "b", "c"])
        assert [n for n, _ in ranked] == ["a", "c", "b"]
        assert ranked[0][1] >= ranked[-1][1]

    def test_explain_headline_structure(self):
        fx = FeatureExtractor()
        corpus = [
            "You Won't Believe This Amazing Trick",
            "Senate Approves Budget Deal",
            "17 Things Only Dog Lovers Know",
            "Market Falls Sharply Amid Fears",
        ]
        scaler = StandardScaler().fit(fx.transform(corpus))
        w = np.ones(fx.n_features)
        exp = interpret.explain_headline(
            fx, scaler, w, 0.0, fx.feature_names,
            "You Won't Believe These 7 Things!", top_k=5,
        )
        assert set(exp) == {"headline", "decision_score", "prediction",
                            "top_contributions", "bias", "other_contribution"}
        assert exp["prediction"] in {"clickbait", "not clickbait"}
        assert len(exp["top_contributions"]) == 5
        for c in exp["top_contributions"]:
            assert set(c) == {"feature", "raw_value", "contribution"}

    @pytest.mark.parametrize("top_k", [0, 3, 32, 50])
    def test_explanation_reconstructs_fitted_model_score(self, top_k):
        fx = FeatureExtractor()
        corpus = [
            "You Won't Believe This Amazing Trick",
            "Senate Approves Budget Deal",
            "17 Things Only Dog Lovers Know",
            "Market Falls Sharply Amid Fears",
        ]
        y = np.array([1, 0, 1, 0])
        scaler = StandardScaler().fit(fx.transform(corpus))
        model = AveragedPerceptron(max_epochs=10).fit(
            scaler.transform(fx.transform(corpus)), y
        )
        assert model.b_ != 0.0  # Exercise the intercept term as well as features.
        headline = "You Won't Believe These 7 Things!"
        exp = interpret.explain_headline(
            fx, scaler, model.w_, model.b_, fx.feature_names, headline, top_k
        )
        x_std = scaler.transform(fx.transform([headline]))
        reconstructed = (
            sum(c["contribution"] for c in exp["top_contributions"])
            + exp["other_contribution"] + exp["bias"]
        )
        assert reconstructed == pytest.approx(exp["decision_score"])
        assert exp["decision_score"] == pytest.approx(model.decision_function(x_std)[0])
        expected_label = "clickbait" if model.predict(x_std)[0] == 1 else "not clickbait"
        assert exp["prediction"] == expected_label
        assert len(exp["top_contributions"]) == min(top_k, fx.n_features)

    def test_weight_stability_reference_signs_and_variation(self):
        folds = np.array([[2.0, -1.0, 0.0], [-2.0, -3.0, 2.0], [4.0, 1.0, 0.0]])
        reference = np.array([1.0, -1.0, 0.0])
        summary = interpret.summarize_weight_stability(folds, ["a", "b", "c"], reference)
        assert [row["feature"] for row in summary] == ["a", "b", "c"]
        np.testing.assert_allclose([row["mean_weight"] for row in summary], folds.mean(axis=0))
        np.testing.assert_allclose([row["std_weight"] for row in summary], folds.std(axis=0))
        assert [row["sign_agreement"] for row in summary] == pytest.approx([2/3, 2/3, 2/3])

    @pytest.mark.parametrize("folds,names,reference", [
        ([], ["a"], [1.0]),
        ([[1.0, 2.0]], ["a"], [1.0]),
        ([[1.0]], ["a", "b"], [1.0]),
        ([[1.0]], ["a"], [[1.0]]),
        ([[np.nan]], ["a"], [1.0]),
        ([[1.0]], ["a"], [np.inf]),
        ([[1.0, 2.0]], ["a", "a"], [1.0, 1.0]),
    ])
    def test_weight_stability_rejects_misaligned_or_nonfinite_inputs(
        self, folds, names, reference
    ):
        with pytest.raises(ValueError):
            interpret.summarize_weight_stability(folds, names, reference)


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
class TestBaseline:
    def test_fit_predict_and_vocab(self):
        corpus = [
            "you won't believe this amazing trick",
            "17 things only dog lovers will understand",
            "which disney princess are you",
            "senate approves budget deal after talks",
            "market falls sharply amid global fears",
            "scientists discover new species of frog",
        ]
        y = np.array([1, 1, 1, 0, 0, 0])
        model = TfidfLogisticRegression(max_features=200).fit(corpus, y)
        preds = model.predict(corpus)
        assert preds.shape == (6,)
        assert set(np.unique(preds)).issubset({0, 1})
        assert model.n_features > 0
        assert model.decision_function(corpus).shape == (6,)


# --------------------------------------------------------------------------- #
# Figure smoke tests                                                          #
# --------------------------------------------------------------------------- #
class TestFigures:
    def test_class_feature_means_are_invariant_to_raw_units(self, monkeypatch, tmp_path):
        import matplotlib.pyplot as plt

        captured = []

        def capture(fig, path):
            ax = fig.axes[0]
            captured.append((
                [tick.get_text() for tick in ax.get_yticklabels()],
                [bar.get_width() for bar in ax.patches],
                ax.get_xlabel(),
            ))
            plt.close(fig)

        monkeypatch.setattr(interpret.ps, "save", capture)
        X = np.array([[1., 10., 7.], [2., 20., 7.], [5., 30., 7.], [6., 40., 7.]])
        y = np.array([0, 0, 1, 1])
        names = ["a", "b", "constant"]
        for values in (X, X * np.array([1000., 0.01, 2.])):
            interpret.plot_class_feature_means(values, y, names, str(tmp_path / "means.png"), top_k=3)
        assert captured[0][0] == captured[1][0]
        np.testing.assert_allclose(captured[0][1], captured[1][1], atol=1e-12)
        assert np.isfinite(captured[0][1]).all()
        assert captured[0][2] == "Mean standardized feature value"

    def test_feature_weights_figure_written(self, tmp_path):
        names = [f"f{i}" for i in range(8)]
        w = np.linspace(-3, 3, 8)
        out = tmp_path / "weights.png"
        interpret.plot_feature_weights(w, names, str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_confusion_matrix_figure_written(self, tmp_path):
        out = tmp_path / "cm.png"
        evaluate.plot_confusion_matrix(
            {"tn": 10, "fp": 2, "fn": 3, "tp": 15}, str(out), "CM"
        )
        assert out.exists() and out.stat().st_size > 0

    def test_convergence_figure_written(self, tmp_path):
        history = {"epoch_updates": [5, 3, 1, 0],
                   "epoch_train_accuracy": [0.8, 0.9, 0.95, 1.0]}
        out = tmp_path / "conv.png"
        evaluate.plot_convergence(history, str(out))
        assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
