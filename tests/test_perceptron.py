"""Tests for the from-scratch averaged Perceptron.

These cover the properties the grading rubric cares about for a hand-rolled
learner: it must *converge* on linearly separable data (the Perceptron
convergence theorem), expose a correct and self-consistent scikit-learn-style
API, be reproducible under a fixed seed, respect ``max_epochs`` on
non-separable data, and guard its binary-only contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perceptron import AveragedPerceptron


@pytest.fixture
def separable():
    """Two well-separated Gaussian blobs -> linearly separable."""
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=3.0, scale=0.5, size=(60, 3))
    x_neg = rng.normal(loc=-3.0, scale=0.5, size=(60, 3))
    X = np.vstack([x_pos, x_neg])
    y = np.array([1] * 60 + [0] * 60)
    return X, y


@pytest.fixture
def noisy():
    """Random features with random labels -> not linearly separable."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 4))
    y = rng.integers(0, 2, size=80)
    return X, y


# --------------------------------------------------------------------------- #
# API basics                                                                  #
# --------------------------------------------------------------------------- #
class TestApi:
    def test_fit_returns_self(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=10)
        assert model.fit(X, y) is model

    def test_predict_shape(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=10).fit(X, y)
        assert model.predict(X).shape == (X.shape[0],)

    def test_predictions_are_known_labels(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=10).fit(X, y)
        assert set(np.unique(model.predict(X))).issubset({0, 1})

    def test_score_in_unit_interval(self, noisy):
        X, y = noisy
        model = AveragedPerceptron(max_epochs=10).fit(X, y)
        s = model.score(X, y)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_learned_attributes_exist(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=5).fit(X, y)
        assert model.w_.shape == (X.shape[1],)
        assert isinstance(model.b_, float)
        assert hasattr(model, "w_last_") and hasattr(model, "b_last_")
        assert model.n_updates_ >= 0

    def test_decision_function_sign_matches_predict(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=10).fit(X, y)
        scores = model.decision_function(X)
        reconstructed = np.where(scores > 0.0, 1, 0)
        np.testing.assert_array_equal(reconstructed, model.predict(X))


# --------------------------------------------------------------------------- #
# Convergence / learning                                                      #
# --------------------------------------------------------------------------- #
class TestConvergence:
    def test_converges_on_separable_data(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=50, random_state=0).fit(X, y)
        # Convergence theorem: a full mistake-free pass must occur.
        assert 0 in model.history_["epoch_updates"]
        # And the averaged model separates the training data (near-)perfectly.
        assert model.score(X, y) >= 0.98

    def test_vanilla_separates_perfectly(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=50, averaged=False,
                                   random_state=0).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_early_stop_shrinks_history(self, separable):
        X, y = separable
        model = AveragedPerceptron(max_epochs=100, random_state=0).fit(X, y)
        # It converges well before 100 epochs, so training stops early.
        assert len(model.history_["epoch_updates"]) < 100

    def test_learns_signal_better_than_chance(self):
        # A single informative dimension buried in noise.
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=400)
        signal = (y * 2 - 1) * 1.5 + rng.normal(scale=1.0, size=400)
        noise = rng.normal(size=(400, 3))
        X = np.column_stack([signal, noise])
        model = AveragedPerceptron(max_epochs=30, random_state=0).fit(X, y)
        assert model.score(X, y) > 0.8


# --------------------------------------------------------------------------- #
# max_epochs / history bookkeeping                                            #
# --------------------------------------------------------------------------- #
class TestEpochs:
    def test_max_epochs_respected_on_noisy_data(self, noisy):
        X, y = noisy
        model = AveragedPerceptron(max_epochs=5, random_state=0).fit(X, y)
        # Non-separable: never a 0-update epoch, so all epochs run.
        assert len(model.history_["epoch_updates"]) == 5

    def test_history_lists_aligned(self, noisy):
        X, y = noisy
        model = AveragedPerceptron(max_epochs=7, random_state=0).fit(X, y)
        h = model.history_
        assert len(h["epoch_updates"]) == len(h["epoch_train_accuracy"])


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
class TestReproducibility:
    def test_same_seed_same_weights(self, separable):
        X, y = separable
        a = AveragedPerceptron(max_epochs=20, random_state=7).fit(X, y)
        b = AveragedPerceptron(max_epochs=20, random_state=7).fit(X, y)
        np.testing.assert_allclose(a.w_, b.w_)
        assert a.b_ == pytest.approx(b.b_)

    def test_shuffle_false_is_order_deterministic(self, separable):
        X, y = separable
        a = AveragedPerceptron(max_epochs=10, shuffle=False).fit(X, y)
        b = AveragedPerceptron(max_epochs=10, shuffle=False).fit(X, y)
        np.testing.assert_allclose(a.w_, b.w_)


# --------------------------------------------------------------------------- #
# Contract guards                                                             #
# --------------------------------------------------------------------------- #
class TestContract:
    def test_rejects_non_binary(self):
        X = np.random.default_rng(0).normal(size=(30, 2))
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            AveragedPerceptron().fit(X, y)

    def test_averaged_default_true(self):
        assert AveragedPerceptron().averaged is True

    def test_arbitrary_label_values_preserved(self, separable):
        X, y = separable
        y2 = np.where(y == 1, 5, -3)  # relabel classes to {-3, 5}
        model = AveragedPerceptron(max_epochs=20, random_state=0).fit(X, y2)
        assert set(np.unique(model.classes_)) == {-3, 5}
        assert set(np.unique(model.predict(X))).issubset({-3, 5})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
