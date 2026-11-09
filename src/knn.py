"""
k-Nearest Neighbours, implemented from scratch with NumPy.

Idea: there is no real "training". We memorise the training set. To classify a
new point we find the k training points closest to it and let them vote.

Design choices exposed as parameters (all investigated in the notebook):
  k        - how many neighbours vote. Small k = very local, noisy decisions
             (low bias, high variance). Large k = smoother decisions that can
             blur class boundaries (high bias, low variance).
  metric   - "euclidean": straight-line distance  ||a - b||
             "cosine":    1 - cos(angle between a and b); ignores vector length
  weights  - "uniform":  every neighbour gets one vote
             "distance": closer neighbours get bigger votes (weight = 1/distance^power)
  power    - only for weights="distance". power=1 is the usual 1/distance. Larger
             powers make the very closest neighbours dominate the vote (power -> infinity
             behaves like 1-NN), while the others still break near-ties.
"""
import numpy as np


class KNNClassifier:
    def __init__(self, k=5, metric="euclidean", weights="uniform", power=1.0, n_classes=None,
                 chunk_size=1000):
        assert metric in ("euclidean", "cosine")
        assert weights in ("uniform", "distance")
        self.k = k
        self.metric = metric
        self.weights = weights
        self.power = power
        self.n_classes = n_classes
        self.chunk_size = chunk_size  # process test points in chunks to limit memory

    # "Training" = remembering the data
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)
        if self.n_classes is None:
            self.n_classes = int(self.y_train.max()) + 1
        if self.metric == "cosine":
            # cosine distance = 1 - (a . b) / (|a| |b|); pre-normalise once
            self._Xn = self.X_train / (np.linalg.norm(self.X_train, axis=1, keepdims=True) + 1e-12)
        else:
            self._sq_norms = (self.X_train ** 2).sum(axis=1)  # |b|^2, reused for every query
        return self

    def _distances(self, Q):
        """Distance from every query row in Q to every training row (matrix)."""
        if self.metric == "cosine":
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            return 1.0 - Qn @ self._Xn.T
        # |a - b|^2 = |a|^2 - 2 a.b + |b|^2   (one matrix multiply instead of a loop)
        d2 = (Q ** 2).sum(axis=1, keepdims=True) - 2.0 * Q @ self.X_train.T + self._sq_norms
        return np.sqrt(np.maximum(d2, 0.0))

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of the k nearest training points, sorted closest first."""
        k = k or self.k
        X = np.asarray(X, dtype=np.float32)
        all_d, all_i = [], []
        for start in range(0, len(X), self.chunk_size):
            D = self._distances(X[start:start + self.chunk_size])
            # argpartition finds the k smallest without fully sorting all 16,000 distances
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            d = np.take_along_axis(D, idx, axis=1)
            order = np.argsort(d, axis=1)
            all_i.append(np.take_along_axis(idx, order, axis=1))
            all_d.append(np.take_along_axis(d, order, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    def vote(self, dist, idx, k=None, weights=None, power=None):
        """
        Turn the neighbour lists into class scores (rows sum to 1).
        Separated from kneighbors() so we can try many k values while computing
        the expensive distances only once.
        """
        k = k or self.k
        weights = weights or self.weights
        power = self.power if power is None else power
        dist, idx = dist[:, :k].astype(np.float64), idx[:, :k]
        labels = self.y_train[idx]                                  # (n, k)
        if weights == "uniform":
            w = np.ones_like(dist)
        else:
            # (closest distance / distance)^power: same ranking as 1/distance^power,
            # but scaled to (0, 1] per row so large powers cannot overflow
            w = ((dist[:, :1] + 1e-8) / (dist + 1e-8)) ** power
        # Tie-break: give closer neighbours a negligible extra weight, so a tied
        # vote goes to the class of the nearest neighbour instead of the lowest label.
        w = w + 1e-6 * (k - np.arange(k))[None, :]
        scores = np.zeros((len(labels), self.n_classes))
        np.add.at(scores, (np.arange(len(labels))[:, None], labels), w)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        d, i = self.kneighbors(X)
        return self.vote(d, i)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
