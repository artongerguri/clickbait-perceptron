"""End-to-end clickbait-detection pipeline.

Run from the project root:

    python main.py                 # full run with defaults
    python main.py --epochs 40     # override training epochs
    python main.py --no-cv         # skip cross-validation (faster)

The script loads the corpus, engineers interpretable features, trains the
from-scratch averaged perceptron, benchmarks it against a black-box
TF-IDF + Logistic Regression baseline, cross-validates, and writes all metrics,
ranked feature weights, per-headline explanations and figures into ``results/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.preprocessing import StandardScaler

from src import evaluate, interpret
from src.baselines import TfidfLogisticRegression
from src.data_loader import dataset_summary, load_dataset, split
from src.features import FeatureExtractor
from src.perceptron import AveragedPerceptron

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

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
    args = parser.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": vars(args)}

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

    # ------------------------------------------------------------------ #
    # 4. Cross-validation                                                #
    # ------------------------------------------------------------------ #
    if not args.no_cv:
        banner(f"4. {args.folds}-fold stratified cross-validation (perceptron)")
        X_all = extractor.transform(headlines)
        cv = evaluate.cross_validate_perceptron(
            X_all, labels,
            model_factory=lambda: AveragedPerceptron(
                max_epochs=args.epochs, random_state=args.seed
            ),
            k=args.folds, random_state=args.seed,
        )
        report["cross_validation"] = cv
        for metric, stats in cv.items():
            print(f"  {metric:<10}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Black-box baseline                                              #
    # ------------------------------------------------------------------ #
    banner("5. Black-box baseline (TF-IDF + Logistic Regression)")
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
    print_metrics("TF-IDF + LogReg (black-box)", base_metrics)

    # ------------------------------------------------------------------ #
    # 6. Head-to-head comparison                                         #
    # ------------------------------------------------------------------ #
    banner("6. Head-to-head comparison")
    print(f"  {'model':<34}{'features':>10}{'accuracy':>11}{'f1':>9}{'roc_auc':>10}")
    for name, m in [
        ("Averaged perceptron (engineered)", perc_metrics),
        ("TF-IDF + LogReg (black-box)", base_metrics),
    ]:
        print(f"  {name:<34}{m['n_features']:>10,}{m['accuracy']:>11.4f}"
              f"{m['f1']:>9.4f}{m['roc_auc']:>10.4f}")

    # ------------------------------------------------------------------ #
    # 7. Figures                                                         #
    # ------------------------------------------------------------------ #
    banner("7. Generating figures")
    evaluate.plot_convergence(perc.history_, str(FIGS / "convergence.png"))
    evaluate.plot_confusion_matrix(
        perc_metrics["confusion_matrix"],
        str(FIGS / "confusion_perceptron.png"),
        "Confusion matrix — averaged perceptron",
    )
    evaluate.plot_confusion_matrix(
        base_metrics["confusion_matrix"],
        str(FIGS / "confusion_baseline.png"),
        "Confusion matrix — TF-IDF + LogReg",
    )
    fpr_p, tpr_p, auc_p = evaluate.roc_points(y_test, y_score)
    fpr_b, tpr_b, auc_b = evaluate.roc_points(y_test, base_score)
    evaluate.plot_roc_curves(
        [
            ("Averaged perceptron", fpr_p, tpr_p, auc_p),
            ("TF-IDF + LogReg", fpr_b, tpr_b, auc_b),
        ],
        str(FIGS / "roc_curves.png"),
    )
    interpret.plot_feature_weights(perc.w_, feature_names, str(FIGS / "feature_weights.png"))
    interpret.plot_class_feature_means(
        X_train, y_train, feature_names, str(FIGS / "class_feature_means.png")
    )
    print(f"  figures written to {FIGS}")

    # ------------------------------------------------------------------ #
    # 8. Interpretability                                                #
    # ------------------------------------------------------------------ #
    banner("8. Interpretability — strongest cues")
    ranked = interpret.rank_weights(perc.w_, feature_names)
    print("  Top clickbait signals:")
    for name, wv in ranked[:8]:
        print(f"    {name:<26} {wv:+.3f}")
    print("  Top legitimate-news signals:")
    for name, wv in ranked[-8:][::-1]:
        print(f"    {name:<26} {wv:+.3f}")

    with open(RESULTS / "feature_weights.csv", "w", encoding="utf-8") as f:
        f.write("feature,weight\n")
        for name, wv in ranked:
            f.write(f"{name},{wv:.6f}\n")

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
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    banner("Done")
    print(f"  metrics  -> {RESULTS / 'metrics.json'}")
    print(f"  weights  -> {RESULTS / 'feature_weights.csv'}")
    print(f"  figures  -> {FIGS}")


def print_metrics(title: str, m: dict) -> None:
    print(f"  [{title}]")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        if key in m:
            print(f"    {key:<10}: {m[key]:.4f}")
    cm = m["confusion_matrix"]
    print(f"    confusion : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


if __name__ == "__main__":
    main()
