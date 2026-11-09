"""
Shared utilities used by every experiment:
  - fixed random seed
  - loading the CSV files (cached as .npz so later loads are fast)
  - ONE stratified train/validation split used by every model
  - preprocessing (fitted on the training split only, to avoid leakage)
  - small helpers for timing and saving results

Every notebook starts with:  from src.common import *
"""
import os
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths and seed
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
PRED_DIR = RESULTS_DIR / "val_predictions"
for _d in (TABLES_DIR, FIG_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_FRACTION = 0.20

N_CLASSES = {"image": 50, "sentiment": 3}
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]


def set_seed(seed=SEED):
    """Seed every random number generator we use (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _read_csv(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None
    ids = df["id"].to_numpy()
    return X, y, ids


def load_task(task):
    """
    Returns X_train, y_train, X_test, test_ids for task in {"image", "sentiment"}.
    The first call parses the CSVs and caches them as data/cache_<task>.npz.
    """
    assert task in ("image", "sentiment")
    cache = DATA_DIR / f"cache_{task}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X_train"], d["y_train"], d["X_test"], d["test_ids"]
    X_train, y_train, _ = _read_csv(f"{task}_train")
    X_test, _, test_ids = _read_csv(f"{task}_test")
    np.savez(cache, X_train=X_train, y_train=y_train, X_test=X_test, test_ids=test_ids)
    return X_train, y_train, X_test, test_ids


# ----------------------------------------------------------------------------
# Validation split (the "practice exam")
# ----------------------------------------------------------------------------
def stratified_split(y, val_fraction=VAL_FRACTION, seed=SEED):
    """
    Returns (train_idx, val_idx). Stratified = every class keeps the same
    proportion in both parts (e.g. 80 of each image class go to validation).
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return train_idx, val_idx


def get_split(task):
    """The single split every model in the project is evaluated on."""
    X, y, X_test, test_ids = load_task(task)
    tr, va = stratified_split(y)
    return X[tr], y[tr], X[va], y[va]


# ----------------------------------------------------------------------------
# Preprocessing. Every option that learns something (mean, std, PCA axes)
# is fitted on the TRAINING split only and then applied to validation/test.
# ----------------------------------------------------------------------------
class Preprocessor:
    """
    mode:
      "none"          - use the features exactly as given
      "standardize"   - per feature: (x - mean) / std     (z-scores)
      "l2"            - scale every row to unit length     (only direction matters)
      "hellinger"     - image only: x / sum(x), then sqrt  (classic trick for
                        bag-of-visual-words histograms; damps very frequent words)
      "hellinger+standardize" - both, in that order
    pca_dims: optional, project onto the top principal components after the above.
    """

    def __init__(self, mode="none", pca_dims=None):
        self.mode = mode
        self.pca_dims = pca_dims

    @staticmethod
    def _hellinger(X):
        s = X.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return np.sqrt(X / s)

    def _base(self, X):
        X = X.astype(np.float32)
        if "hellinger" in self.mode:
            X = self._hellinger(X)
        if self.mode == "l2":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X

    def fit(self, X):
        Xb = self._base(X)
        if "standardize" in self.mode:
            self.mean_ = Xb.mean(axis=0)
            self.std_ = Xb.std(axis=0) + 1e-8
            Xb = (Xb - self.mean_) / self.std_
        if self.pca_dims:
            # PCA via SVD of the centred training matrix
            self.pca_mean_ = Xb.mean(axis=0)
            _, S, Vt = np.linalg.svd(Xb - self.pca_mean_, full_matrices=False)
            self.components_ = Vt[: self.pca_dims]
            self.explained_var_ratio_ = (S ** 2 / (S ** 2).sum())[: self.pca_dims]
        return self

    def transform(self, X):
        Xb = self._base(X)
        if "standardize" in self.mode:
            Xb = (Xb - self.mean_) / self.std_
        if self.pca_dims:
            Xb = (Xb - self.pca_mean_) @ self.components_.T
        return Xb.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


class Timer:
    """with Timer() as t: ...   then t.seconds"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def save_table(df, name):
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_val_predictions(task, model_name, y_pred, proba=None, extra=None):
    """Saved so the error-analysis notebook can compare models' mistakes."""
    np.savez(PRED_DIR / f"{task}_{model_name}.npz", y_pred=y_pred,
             proba=proba if proba is not None else np.array([]))
    if extra is not None:
        with open(PRED_DIR / f"{task}_{model_name}.json", "w") as f:
            json.dump(extra, f, indent=2)


def save_json(obj, name):
    with open(TABLES_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
