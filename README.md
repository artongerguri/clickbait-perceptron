<p align="center">
  <img src="assets/up-logo.png" alt="Logo e Universitetit të Prishtinës" width="180">
</p>

<h1 align="center">Universiteti i Prishtinës</h1>

<h2 align="center">Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</h2>

<p align="center">
  Programi i Studimeve: Inxhinieri Kompjuterike dhe Softuerike – Master<br>
  Lënda: Procesimi i Gjuhëve Natyrale (NLP)<br>
  Viti akademik: 2026/27
</p>

<br>

<h1 align="center">Perceptron-Based Text Classification for Detecting Clickbait Headlines</h1>

<br>

<h2 align="center">Profesori</h2>

<p align="center">Prof. Asst. Dr. Mërgim H. Hoti</p>

<h2 align="center">Studentët</h2>

<p align="center">
  Arton Gërguri<br>
  Jon Klinaku
</p>

<br>

---

## Abstrakt

Titujt *clickbait* (në shqip «karrem klikimesh» — tituj të formuluar posaçërisht për të zgjuar kuriozitet dhe për të nxitur klikimin, shpesh pa e përmbushur premtimin) shfrytëzojnë kuriozitetin e lexuesit për të maksimizuar klikimet pa dhënë informacion proporcional. Ndryshe nga detektorët neuralë "black-box", ky projekt ndjek një qasje të interpretueshme: një perceptron i mesatarizuar i implementuar nga e para (NumPy) ushqehet me 33 veçori leksikore dhe stilistike të dizajnuara me dorë, secila një tregues gjuhësor i motivuar — numra listicle, adresim i lexuesit, fraza kurioziteti, strukturë pyetëse, hiperbolë, pikësim, stil i shkronjave të mëdha dhe afekt. Në korpusin *Stop Clickbait* me 32,000 tituj, modeli arrin 97.56% saktësi dhe 0.9971 ROC-AUC, duke tejkaluar një bazë black-box TF-IDF + Regresion Logjistik (96.89%) me rreth 600× më pak veçori. Meqë modeli është linear mbi veçori të standardizuara dhe të emërtuara, çdo peshë është një masë e drejtpërdrejtë se *cili tregues gjuhësor sinjalizon clickbait*.

| Modeli | Veçori | Saktësia | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Perceptron i mesatarizuar (ynë) | 33 | 0.9756 | 0.9777 | 0.9734 | 0.9756 | 0.9971 |
| TF-IDF + Regresion Logjistik | 20,000 | 0.9689 | 0.9741 | 0.9634 | 0.9687 | 0.9957 |

---

## Motivation

*Clickbait* headlines open a "curiosity gap" to maximize clicks regardless of how
much information they actually deliver. Most modern detectors are accurate but
opaque neural networks. This project takes the opposite, interpretability-first
stance: how far can a *linear* model driven by a small set of transparent,
linguistically motivated features go? The answer here is: slightly further than a
strong black box — and it can explain itself.

## Dataset

The *Stop Clickbait* corpus (Chakraborty et al., 2016): 32,000 English
headlines, each labelled `1` (clickbait) or `0` (not clickbait).

| Property | Value |
|---|---|
| Total headlines | 32,000 |
| Clickbait / Not | 15,999 / 16,001 (balanced) |
| Avg. words/headline | 9.07 (range 1–26) |
| Train / Test split | 25,600 / 6,400 (stratified, seed 42) |

Clickbait examples come from sources such as BuzzFeed and Upworthy; non-clickbait
from mainstream agencies (NYT, The Guardian, Reuters/WikiNews). The CSV lives at
[`data/clickbait_data.csv`](data/clickbait_data.csv).

## Engineered features

33 named features across 12 linguistic families — the interpretable core
([`src/features.py`](src/features.py)):

| Family | Example features |
|---|---|
| Numeric / listicle | `starts_with_number`, `contains_year` |
| Reader address | `second_person_count`, `first_person_count` |
| Forward reference | `demonstrative_count`, `has_forward_reference` |
| Clickbait phrases | `clickbait_phrase_count` |
| Interrogative | `is_question`, `starts_with_question_word` |
| Hyperbole | `superlative_count`, `intensifier_count` |
| Punctuation | `exclamation_count`, `has_ellipsis` |
| Length | `word_count`, `avg_word_length` |
| Capitalization | `title_case_ratio`, `all_caps_ratio` |
| Informal register | `slang_count` |
| Affect | `positive/negative/emotional_word_count` |
| Function words | `stopword_ratio` |

Features are standardized before training so weight magnitudes are directly
comparable for interpretation (the scaler is fit on the training fold only, to
avoid leakage).

## Model: averaged perceptron (from scratch)

[`src/perceptron.py`](src/perceptron.py) implements the averaged perceptron
(Freund & Schapire, 1999) in NumPy, with:

- the efficient cached-accumulator averaging trick (no storing of every iterate),
- convergence detection (a full mistake-free pass),
- a scikit-learn-style API (`fit` / `predict` / `decision_function` / `score`).

Averaging reduces variance and improves generalization at no inference cost, while
keeping the model linear and interpretable.

## Results

5-fold stratified cross-validation (perceptron): accuracy 0.9764 ± 0.0017,
ROC-AUC 0.9973 ± 0.0003 — stable and leak-free (scaler re-fit per fold).

Implementation verification (test accuracy): vanilla `0.9700` · averaged `0.9756` · scikit-learn `Perceptron` `0.9670`.

<div align="center">

<img src="results/figures/roc_curves.png" alt="ROC curves" width="47%">
<img src="results/figures/confusion_perceptron.png" alt="Confusion matrix" width="42%">

<img src="results/figures/convergence.png" alt="Convergence" width="90%">

</div>

## Interpretability — which cues signal clickbait?

Because the classifier is linear over standardized named features, its weights rank
the cues ([`src/interpret.py`](src/interpret.py), full list in
[`results/feature_weights.csv`](results/feature_weights.csv)):

| Strongest clickbait cues | Weight | Strongest legitimate cues | Weight |
|---|---:|---|---:|
| `title_case_ratio` | +117.6 | `is_question` | −17.6 |
| `clickbait_phrase_count` | +37.2 | `has_clickbait_phrase` | −17.4 |
| `starts_with_auxiliary` | +21.5 | `char_count` | −8.1 |
| `double_quote_count` | +17.9 | `all_caps_word_count` | −4.1 |
| `slang_count` | +16.5 | `has_forward_reference` | −3.0 |

<div align="center">
<img src="results/figures/feature_weights.png" alt="Feature weights" width="78%">
</div>

Individual predictions are explained by decomposing the score into per-feature
contributions (`contribution = standardized_value × weight`); see the case studies
printed by `main.py` and stored in [`results/metrics.json`](results/metrics.json).

## Project structure

```text
clickbait-perceptron/
├── assets/
│   └── up-logo.png                # university logo
├── data/
│   └── clickbait_data.csv          # Stop Clickbait corpus (32k headlines)
├── docs/
│   ├── paper.html                 # IEEE-style paper with print layout
│   └── Perceptron-Based Text Classification for Detecting Clickbait Headlines.pdf
├── results/                       # saved experiment outputs
│   ├── feature_weights.csv        # ranked perceptron feature weights
│   ├── metrics.json               # configuration, metrics, and explanations
│   ├── run_log.txt                # saved console output from a pipeline run
│   └── figures/
│       ├── class_feature_means.png # feature averages by class
│       ├── confusion_baseline.png
│       ├── confusion_perceptron.png
│       ├── convergence.png         # training progress
│       ├── feature_weights.png
│       └── roc_curves.png          # perceptron and baseline comparison
├── src/
│   ├── __init__.py                # source package marker
│   ├── baselines.py               # TF-IDF + Logistic Regression baseline
│   ├── data_loader.py             # CSV loading and stratified train/test split
│   ├── evaluate.py                # metrics, cross-validation, evaluation plots
│   ├── features.py                # 33 interpretable linguistic features
│   ├── interpret.py               # weight ranking and headline explanations
│   ├── perceptron.py              # vanilla and averaged perceptron in NumPy
│   ├── plotting_style.py          # shared, colorblind-safe figure style
│   └── preprocessing.py           # casing/punctuation-preserving cleaning
├── tests/
│   ├── __init__.py                # test package marker
│   ├── conftest.py                # project import path setup for pytest
│   ├── test_features.py           # feature extraction and edge cases
│   ├── test_perceptron.py         # training, predictions, and model contract
│   ├── test_pipeline.py           # evaluation, interpretation, baseline, plots
│   └── test_preprocessing.py      # text cleaning and tokenization
├── .gitignore                     # Python, environment, and editor exclusions
├── LICENSE                        # MIT license
├── main.py                        # command-line entry point for the pipeline
├── README.md                      # project overview and usage
└── requirements.txt               # runtime and test dependencies
```

[`main.py`](main.py) loads the dataset from `data/`, runs the components in
`src/`, and writes metrics, feature weights, and figures to `results/`.
The saved results are tracked in the repository; `docs/` contains the paper's
HTML source and PDF version.

## Installation & usage

```bash
# 1. (optional) virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. dependencies
pip install -r requirements.txt

# 3. run the full pipeline (train, evaluate, compare, explain, plot)
python main.py

# faster run without cross-validation
python main.py --no-cv
```

Outputs are written to `results/` (metrics, ranked weights, and all figures).
Useful flags: `--epochs`, `--test-size`, `--seed`, `--folds`.

## Testing & verification

```bash
pip install pytest
python -m pytest -q          # 85 tests
```

Tests cover perceptron convergence on separable data, the binary-only contract,
reproducibility, every feature extractor, degenerate-headline edge cases, metrics,
leak-free cross-validation, the baseline, and the figure code paths.

## Limitations

- `title_case_ratio` partly reflects a house-style artifact of the corpus
  (clickbait sources capitalize every word; agencies use sentence case), so the
  model may learn source style as well as intrinsic clickbaitiness — cross-source
  evaluation is needed before deployment.
- Lexicons are English- and period-specific; features are headline-only (no article
  body). These are discussed in §VIII of the paper.

## References

1. A. Chakraborty, B. Paranjape, S. Kakarla, N. Ganguly. *Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media.* ASONAM, 2016.
2. F. Rosenblatt. *The perceptron: a probabilistic model for information storage and organization in the brain.* Psychological Review, 1958.
3. Y. Freund, R. E. Schapire. *Large margin classification using the perceptron algorithm.* Machine Learning, 1999.
4. M. Collins. *Discriminative training methods for HMMs: perceptron algorithms.* EMNLP, 2002.
5. F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011.

## License

Released under the [MIT License](LICENSE). Dataset © its original authors
(Chakraborty et al., 2016), used here for academic purposes.
