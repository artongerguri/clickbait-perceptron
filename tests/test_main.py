"""Exercise report generation and keep holdout headlines out of validation."""

import csv
import json
import sys

import numpy as np

import main
from src.data_loader import split
from src.features import FeatureExtractor


def test_pipeline_keeps_holdout_out_of_cv_and_writes_reports(monkeypatch, tmp_path):
    headlines = np.array([
        f"You Won't Believe These {i} Amazing Tricks" for i in range(12)
    ] + [f"Council approves budget measure {i} after debate" for i in range(12)])
    labels = np.array([1] * 12 + [0] * 12)
    train, test, _, _ = split(headlines, labels, test_size=0.25, random_state=42)
    monkeypatch.setattr(main, "load_dataset", lambda: (headlines, labels))
    actual_cv = main.compare_cross_validation

    def checked_cv(cv_headlines, cv_labels, **kwargs):
        assert set(cv_headlines) == set(train)
        assert not set(cv_headlines).intersection(test)
        return actual_cv(cv_headlines, cv_labels, **kwargs)

    monkeypatch.setattr(main, "compare_cross_validation", checked_cv)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--epochs", "2", "--folds", "2", "--test-size", "0.25",
        "--output-dir", str(tmp_path),
    ])
    main.main()

    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["cross_validation"]["n_samples"] == len(train)
    assert report["protocol"]["test_samples"] == len(test)
    assert report["engineered_logreg"]["n_features"] == FeatureExtractor().n_features
    assert len(report["weight_stability"]) == FeatureExtractor().n_features
    assert len(report["capitalization_robustness"]["variants"]) == 3
    original = report["capitalization_robustness"]["variants"]["full"]["conditions"]["original"]
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
        assert original[metric] == report["averaged_perceptron"][metric]
    assert len(list((tmp_path / "figures").glob("*.png"))) == 8
    for name, expected_rows in [
        ("model_comparison.csv", 3), ("capitalization_robustness.csv", 9),
        ("weight_stability.csv", FeatureExtractor().n_features),
    ]:
        with (tmp_path / name).open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == expected_rows

    def unexpected_cv(*args, **kwargs):
        raise AssertionError("--no-cv must skip cross-validation")

    monkeypatch.setattr(main, "compare_cross_validation", unexpected_cv)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--no-cv"])
    main.main()
    no_cv_report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "cross_validation" not in no_cv_report
    assert "weight_stability" not in no_cv_report
    assert not (tmp_path / "weight_stability.csv").exists()
