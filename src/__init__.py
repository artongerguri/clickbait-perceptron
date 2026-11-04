"""Clickbait Perceptron — interpretable, feature-based clickbait headline detection.

Package modules
---------------
- preprocessing : lightweight tokenization/normalization helpers.
- features      : engineered lexical & stylistic feature extraction (the core).
- perceptron    : an averaged Perceptron implemented from scratch with NumPy.
- baselines     : a black-box TF-IDF + Logistic Regression reference model.
- data_loader   : dataset loading and reproducible train/test splitting.
- evaluate      : classification metrics, confusion matrix and plots.
- interpret     : analysis and visualization of learned feature weights.
"""

__version__ = "1.0.0"
