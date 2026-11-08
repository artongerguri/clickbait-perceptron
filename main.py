"""End-to-end clickbait-detection pipeline.

Run from the project root:

    python main.py                 # full run with defaults
    python main.py --epochs 40     # override training epochs
    python main.py --no-cv         # skip cross-validation (faster)

The script compares the from-scratch averaged perceptron with logistic regression
on the same engineered features and a TF-IDF linear baseline. It cross-validates
on the training partition, measures capitalization sensitivity, and writes
metrics, coefficient summaries, explanations and figures into ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron as SklearnPerceptron
from sklearn.preprocessing import StandardScaler

# This command-line pipeline writes figures to files and needs no GUI toolkit.
matplotlib.use("Agg")

from src import evaluate, interpret
from src.baselines import TfidfLogisticRegression
from src.data_loader import DEFAULT_PATH, dataset_summary, load_dataset, split
from src.experiments import (
    compare_cross_validation,
    plot_capitalization_robustness,
    run_capitalization_experiments,
)
from src.features import FeatureExtractor
from src.perceptron import AveragedPerceptron

ROOT = Path(__file__).resolve().parent

# A few illustrative headlines for the interpretability case study.
CASE_STUDY = [
    "17 Things Only People Who Love Coffee Will Understand",
    "You Won't Believe What This Dog Did Next",
    "Which Disney Princess Are You Based On Your Zodiac Sign",
    "Senate Approves Budget Deal After Months of Negotiation",
    "Scientists Discover New Species of Frog in the Amazon",
    "Stocks Fall Sharply as Oil Prices Surge",
]


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clickbait perceptron pipeline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    parser.add_argument("--output-dir", default="results", help="output directory (relative to project root)")
    args = parser.parse_args()

    results = Path(args.output_dir)
    if not results.is_absolute():
        results = ROOT / results
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": vars(args), "provenance": provenance()}

    # ------------------------------------------------------------------ #
    # 1. Data                                                            #
    # ------------------------------------------------------------------ #
    banner("1. Loading dataset")
    headlines, labels = load_dataset()
    summary = dataset_summary(headlines, labels)
    report["dataset"] = summary
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")

    X_train_txt, X_test_txt, y_train, y_test = split(
        headlines, labels, test_size=args.test_size, random_state=args.seed
    )
    print(f"  train/test sizes   : {len(X_train_txt)} / {len(X_test_txt)}")
    report["protocol"] = {
        "train_samples": len(X_train_txt),
        "test_samples": len(X_test_txt),
        "validation_scope": "Training partition only; the test partition is excluded from all CV folds.",
        "comparison": "Fixed settings, paired validation folds, and identical engineered inputs for both linear classifiers.",
        "scope": "Exploratory in-corpus benchmark; no external or source-held-out validation.",
    }

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                             #
    # ------------------------------------------------------------------ #
    banner("2. Engineering interpretable features")
    extractor = FeatureExtractor()
    feature_names = extractor.feature_names
    print(f"  {extractor.n_features} engineered features:")
    print("   ", ", ".join(feature_names))

    t0 = time.perf_counter()
    X_train = extractor.transform(X_train_txt)
    X_test = extractor.transform(X_test_txt)
    feat_time = time.perf_counter() - t0
    print(f"  feature extraction : {feat_time:.2f}s "
          f"({feat_time / len(headlines) * 1e6:.1f} µs/headline)")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 3. From-scratch averaged perceptron                                #
    # ------------------------------------------------------------------ #
    banner("3. Training averaged perceptron (from scratch)")
    perc = AveragedPerceptron(max_epochs=args.epochs, random_state=args.seed, verbose=True)
    t0 = time.perf_counter()
    perc.fit(X_train_std, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = perc.predict(X_test_std)
    infer_time = time.perf_counter() - t0
    y_score = perc.decision_function(X_test_std)

    perc_metrics = evaluate.compute_metrics(y_test, y_pred, y_score)
    perc_metrics["train_time_s"] = round(train_time, 3)
    perc_metrics["inference_time_s"] = round(infer_time, 4)
    perc_metrics["n_updates"] = int(perc.n_updates_)
    perc_metrics["n_features"] = int(extractor.n_features)
    report["averaged_perceptron"] = perc_metrics
    print_metrics("Averaged perceptron (engineered features)", perc_metrics)

    # 3b. Verification: vanilla vs averaged, and vs scikit-learn's Perceptron.
    banner("3b. Verification against variants & scikit-learn")
    vanilla = AveragedPerceptron(max_epochs=args.epochs, averaged=False,
                                 random_state=args.seed).fit(X_train_std, y_train)
    sk = SklearnPerceptron(max_iter=args.epochs, random_state=args.seed, tol=None)
    sk.fit(X_train_std, y_train)
    report["verification"] = {
        "vanilla_perceptron_acc": round(vanilla.score(X_test_std, y_test), 4),
        "averaged_perceptron_acc": round(perc.score(X_test_std, y_test), 4),
        "sklearn_perceptron_acc": round(float(sk.score(X_test_std, y_test)), 4),
    }
    for k, v in report["verification"].items():
        print(f"  {k:<28}: {v}")

    banner("3c. Logistic regression on the same engineered features")
    control = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=args.seed)
    t0 = time.perf_counter()
    control.fit(X_train_std, y_train)
    control_train_time = time.perf_counter() - t0
    control_metrics = evaluate.compute_metrics(
        y_test, control.predict(X_test_std), control.decision_function(X_test_std)
    )
    control_metrics["train_time_s"] = round(control_train_time, 3)
    control_metrics["n_features"] = extractor.n_features
    report["engineered_logreg"] = control_metrics
    print_metrics("Logistic regression (same engineered features)", control_metrics)

    # ------------------------------------------------------------------ #
    # 4. Cross-validation                                                #
    # ------------------------------------------------------------------ #
    if not args.no_cv:
        banner(f"4. Paired {args.folds}-fold cross-validation (training partition only)")
        cv = compare_cross_validation(
            X_train_txt, y_train, epochs=args.epochs, folds=args.folds, seed=args.seed
        )
        report["weight_stability"] = interpret.summarize_weight_stability(
            cv.pop("perceptron_fold_weights"), feature_names, perc.w_
        )
        report["cross_validation"] = cv
        for name, metrics in cv["models"].items():
            acc = metrics["accuracy"]
            auc = metrics["roc_auc"]
            print(f"  {name:<25} accuracy {acc['mean']:.4f} ± {acc['std']:.4f}"
                  f"  AUC {auc['mean']:.4f} ± {auc['std']:.4f}")
        write_csv(results / "weight_stability.csv", report["weight_stability"])
    else:
        # A reused output directory must not retain stability data from an older run.
        (results / "weight_stability.csv").unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 5. High-dimensional linear baseline                                #
    # ------------------------------------------------------------------ #
    banner("5. TF-IDF + Logistic Regression (linear text baseline)")
    base = TfidfLogisticRegression(random_state=args.seed)
    t0 = time.perf_counter()
    base.fit(X_train_txt, y_train)
    base_train_time = time.perf_counter() - t0
    base_pred = base.predict(X_test_txt)
    base_score = base.decision_function(X_test_txt)
    base_metrics = evaluate.compute_metrics(y_test, base_pred, base_score)
    base_metrics["train_time_s"] = round(base_train_time, 3)
    base_metrics["n_features"] = int(base.n_features)
    report["tfidf_logreg_baseline"] = base_metrics
    print_metrics("TF-IDF + LogReg (linear text baseline)", base_metrics)

    # ------------------------------------------------------------------ #
    # 6. Head-to-head comparison                                         #
    # ------------------------------------------------------------------ #
    banner("6. Head-to-head comparison")
    print(f"  {'model':<34}{'features':>10}{'accuracy':>11}{'f1':>9}{'roc_auc':>10}")
    comparisons = [
        ("Averaged perceptron (engineered)", perc_metrics),
        ("LogReg (same engineered features)", control_metrics),
        ("TF-IDF + LogReg", base_metrics),
    ]
    for name, m in comparisons:
        print(f"  {name:<34}{m['n_features']:>10,}{m['accuracy']:>11.4f}"
              f"{m['f1']:>9.4f}{m['roc_auc']:>10.4f}")
    write_csv(results / "model_comparison.csv", [
        {"model": name, **{key: metrics[key] for key in
         ("n_features", "accuracy", "precision", "recall", "f1", "roc_auc")}}
        for name, metrics in comparisons
    ])

    banner("6b. Capitalization ablation and frozen-model stress tests")
    robustness = run_capitalization_experiments(
        X_train_txt, y_train, X_test_txt, y_test, epochs=args.epochs, seed=args.seed
    )
    report["capitalization_robustness"] = robustness
    robustness_rows = []
    for name, variant in robustness["variants"].items():
        print(f"  {name} ({variant['n_features']} features)")
        for condition, metrics in variant["conditions"].items():
            print(f"    {condition:<20}: accuracy {metrics['accuracy']:.4f}")
            robustness_rows.append({
                "variant": name, "n_features": variant["n_features"], "condition": condition,
                **{key: metrics[key] for key in
                   ("accuracy", "precision", "recall", "f1", "roc_auc", "prediction_changes_vs_original")},
            })
    write_csv(results / "capitalization_robustness.csv", robustness_rows)

    # ------------------------------------------------------------------ #
    # 7. Figures                                                         #
    # ------------------------------------------------------------------ #
    banner("7. Generating figures")
    evaluate.plot_convergence(perc.history_, str(figs / "convergence.png"))
    evaluate.plot_confusion_matrix(
        perc_metrics["confusion_matrix"],
        str(figs / "confusion_perceptron.png"),
        "Confusion matrix — averaged perceptron",
    )
    evaluate.plot_confusion_matrix(
        base_metrics["confusion_matrix"],
        str(figs / "confusion_baseline.png"),
        "Confusion matrix — TF-IDF + LogReg",
    )
    evaluate.plot_confusion_matrix(
        control_metrics["confusion_matrix"], str(figs / "confusion_engineered_logreg.png"),
        "Confusion matrix - logistic regression (engineered)",
    )
    fpr_p, tpr_p, auc_p = evaluate.roc_points(y_test, y_score)
    fpr_b, tpr_b, auc_b = evaluate.roc_points(y_test, base_score)
    fpr_c, tpr_c, auc_c = evaluate.roc_points(y_test, control.decision_function(X_test_std))
    evaluate.plot_roc_curves(
        [
            ("Averaged perceptron", fpr_p, tpr_p, auc_p),
            ("LogReg (engineered)", fpr_c, tpr_c, auc_c),
            ("TF-IDF + LogReg", fpr_b, tpr_b, auc_b),
        ],
        str(figs / "roc_curves.png"),
    )
    interpret.plot_feature_weights(perc.w_, feature_names, str(figs / "feature_weights.png"))
    interpret.plot_class_feature_means(
        X_train, y_train, feature_names, str(figs / "class_feature_means.png")
    )
    plot_capitalization_robustness(robustness, str(figs / "capitalization_robustness.png"))
    print(f"  figures written to {figs}")

    # ------------------------------------------------------------------ #
    # 8. Interpretability                                                #
    # ------------------------------------------------------------------ #
    banner("8. Conditional model coefficients (correlated cues can share effects)")
    ranked = interpret.rank_weights(perc.w_, feature_names)
    print("  Most positive model coefficients:")
    for name, wv in ranked[:8]:
        print(f"    {name:<26} {wv:+.3f}")
    print("  Most negative model coefficients:")
    for name, wv in ranked[-8:][::-1]:
        print(f"    {name:<26} {wv:+.3f}")

    write_csv(results / "feature_weights.csv", [
        {"feature": name, "weight": float(weight)} for name, weight in ranked
    ])
    report["feature_names"] = feature_names
    report["interpretation_note"] = (
        "Coefficients describe this fitted model, conditional on its other inputs; "
        "correlated features can redistribute weights. Fold sign agreement is "
        "descriptive, not a causal effect or confidence interval."
    )

    banner("8b. Per-headline explanations (case study)")
    explanations = []
    for h in CASE_STUDY:
        exp = interpret.explain_headline(
            extractor, scaler, perc.w_, float(perc.b_), feature_names, h
        )
        explanations.append(exp)
        print(f"\n  \"{h}\"")
        print(f"    -> {exp['prediction']}  (score {exp['decision_score']:+.3f})")
        for c in exp["top_contributions"]:
            print(f"       {c['feature']:<24} value={c['raw_value']:<6g} "
                  f"contrib={c['contribution']:+.3f}")
    report["interpretability_examples"] = explanations

    # ------------------------------------------------------------------ #
    # 9. Persist report                                                  #
    # ------------------------------------------------------------------ #
    with open(results / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    banner("Done")
    print(f"  metrics  -> {results / 'metrics.json'}")
    print(f"  weights  -> {results / 'feature_weights.csv'}")
    print(f"  figures  -> {figs}")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Persist a nonempty report table with stable field ordering."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def provenance() -> dict:
    """Record data/source fingerprints and the versions used for this run."""
    source = hashlib.sha256()
    for path in [ROOT / "main.py", *sorted((ROOT / "src").glob("*.py"))]:
        source.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        source.update(b"\0")
        source.update(path.read_bytes())
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
        "dataset_sha256": hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest(),
        "source_sha256": source.hexdigest(),
    }


def print_metrics(title: str, m: dict) -> None:
    print(f"  [{title}]")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in m:
            print(f"    {key:<10}: {m[key]:.4f}")
    cm = m["confusion_matrix"]
    print(f"    confusion : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


if __name__ == "__main__":
    main()
