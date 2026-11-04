"""Dataset loading and reproducible splitting.

Dataset
-------
`clickbait_data.csv` is the balanced **"Stop Clickbait"** corpus of Chakraborty
et al. (2016): 32,000 English headlines, each labelled ``1`` (clickbait) or
``0`` (not clickbait), with the two classes almost perfectly balanced (~16k /
~16k). The clickbait side was collected from sites such as BuzzFeed and
Upworthy; the non-clickbait side from mainstream outlets such as The New York
Times, The Guardian and Reuters wikinews.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HEADLINE_COL = "headline"
LABEL_COL = "clickbait"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "clickbait_data.csv"


def load_dataset(path: str | Path = DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load headlines and labels from the CSV.

    Returns
    -------
    headlines : np.ndarray[str]
    labels    : np.ndarray[int]  (0/1)
    """
    df = pd.read_csv(path)
    missing = {HEADLINE_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")

    # Drop rows with an empty/NaN headline; keep everything else intact.
    df = df.dropna(subset=[HEADLINE_COL]).copy()
    df[HEADLINE_COL] = df[HEADLINE_COL].astype(str)
    df = df[df[HEADLINE_COL].str.strip() != ""]

    headlines = df[HEADLINE_COL].to_numpy()
    labels = df[LABEL_COL].astype(int).to_numpy()
    return headlines, labels


def dataset_summary(headlines: np.ndarray, labels: np.ndarray) -> dict:
    """Compute descriptive statistics used in the EDA / documentation."""
    word_counts = np.array([len(str(h).split()) for h in headlines])
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return {
        "n_samples": int(len(headlines)),
        "n_clickbait": pos,
        "n_non_clickbait": neg,
        "clickbait_ratio": round(pos / len(headlines), 4),
        "avg_words": round(float(word_counts.mean()), 2),
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
    }


def split(
    headlines: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split (preserves the class balance)."""
    return train_test_split(
        headlines,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
