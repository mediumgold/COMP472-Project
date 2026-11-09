"""
Random Forest, implemented from scratch on top of our own DecisionTreeClassifier.

Why it works: a single deep tree has low bias but very high variance (small
changes in the data give a very different tree). A forest trains many trees
that are made deliberately different, then averages their predictions, which
cancels much of that variance. Two sources of randomness:
  1. Bootstrap sampling: each tree trains on n rows drawn WITH replacement
     (~63% unique rows; the other ~37% are that tree's "out-of-bag" rows).
  2. Feature subsampling: at every node only `max_features` random features
     are considered (default sqrt(d)), so trees don't all pick the same splits.

Prediction = average of the trees' class-probability vectors ("soft voting").

Out-of-bag (OOB) score: each training row is predicted only by the trees that
did NOT see it. That gives a free accuracy estimate without a validation set.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .decision_tree import Binner, DecisionTreeClassifier

# --- helpers for building trees in parallel worker processes (n_jobs > 1) ---
_SHARED = {}


def _init_worker(Xb, y, binner, tree_params, n_classes, bootstrap):
    _SHARED.update(Xb=Xb, y=y, binner=binner, tree_params=tree_params,
                   n_classes=n_classes, bootstrap=bootstrap)


def _build_tree(seeds):
    """Grow one tree. Its bootstrap sample and feature choices come only from
    `seeds`, so results are identical whether we use 1 process or many."""
    S = _SHARED
    n = len(S["y"])
    rng = np.random.default_rng(int(seeds[0]))
    idx = rng.integers(0, n, size=n) if S["bootstrap"] else np.arange(n)
    tree = DecisionTreeClassifier(**S["tree_params"], random_state=int(seeds[1]))
    tree.fit_binned(S["Xb"], S["y"], S["binner"], idx, S["n_classes"])
    return tree, idx


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", criterion="gini",
                 bootstrap=True, n_bins=32, oob_score=True, random_state=None,
                 n_jobs=1, verbose=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.n_bins = n_bins
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs          # -1 = use all CPU cores
        self.verbose = verbose

    def fit(self, X, y, n_classes=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        self.n_classes_ = int(n_classes or (y.max() + 1))
        rng = np.random.default_rng(self.random_state)

        # Bin once, share between all trees (each tree only gets different row indices)
        self.binner_ = Binner(self.n_bins).fit(X)
        Xb = self.binner_.transform(X)

        tree_params = dict(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
                           max_features=self.max_features, n_bins=self.n_bins)
        seeds = rng.integers(2 ** 31, size=(self.n_estimators, 2))
        n_jobs = os.cpu_count() if self.n_jobs == -1 else max(1, self.n_jobs)
        init_args = (Xb, y, self.binner_, tree_params, self.n_classes_, self.bootstrap)
        if n_jobs == 1:
            _init_worker(*init_args)
            results = map(_build_tree, seeds)
        else:
            pool = ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=init_args)
            results = pool.map(_build_tree, seeds)

        self.trees_ = []
        oob_sum = np.zeros((n, self.n_classes_))
        oob_cnt = np.zeros(n)
        for t, (tree, idx) in enumerate(results):
            self.trees_.append(tree)
            if self.oob_score and self.bootstrap:
                oob = np.ones(n, dtype=bool)
                oob[idx] = False
                oob_sum[oob] += tree.predict_proba(X[oob])
                oob_cnt[oob] += 1
            if self.verbose and (t + 1) % 25 == 0:
                print(f"  tree {t + 1}/{self.n_estimators}")
        if n_jobs > 1:
            pool.shutdown()

        if self.oob_score and self.bootstrap:
            seen = oob_cnt > 0
            self.oob_score_ = float(np.mean(oob_sum[seen].argmax(1) == y[seen]))
        imp = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        return self

    def staged_predict_proba(self, X):
        """Yield the forest's prediction after 1, 2, ..., n_estimators trees."""
        total = np.zeros((len(X), self.n_classes_))
        for t, tree in enumerate(self.trees_, start=1):
            total += tree.predict_proba(X)
            yield total / t

    def predict_proba(self, X):
        total = np.zeros((len(X), self.n_classes_))
        for tree in self.trees_:
            total += tree.predict_proba(X)
        return total / len(self.trees_)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
