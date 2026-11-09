"""
k-Means clustering, implemented from scratch with NumPy (unsupervised: labels are never used).

Algorithm (Lloyd's algorithm):
  1. Pick k starting centres (k-means++: spread out, each new centre chosen
     with probability proportional to its squared distance from existing centres).
  2. Assign every point to its nearest centre.
  3. Move each centre to the mean of the points assigned to it.
  4. Repeat 2-3 until assignments stop changing.
It minimises the "inertia" = sum of squared distances from points to their centre.
Because the result depends on the starting centres, we run it n_init times and
keep the run with the lowest inertia.
"""
import numpy as np


def _sq_dists(X, C, X_sq=None):
    X_sq = (X ** 2).sum(1)[:, None] if X_sq is None else X_sq
    return np.maximum(X_sq - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0.0)


class KMeans:
    def __init__(self, n_clusters=8, n_init=5, max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_pp(self, X, X_sq, rng):
        n = len(X)
        centres = [X[rng.integers(n)]]
        d2 = _sq_dists(X, np.array(centres), X_sq)[:, 0]
        for _ in range(1, self.n_clusters):
            probs = d2 / d2.sum()
            centres.append(X[rng.choice(n, p=probs)])
            d2 = np.minimum(d2, _sq_dists(X, centres[-1][None, :], X_sq)[:, 0])
        return np.array(centres)

    def _single_run(self, X, X_sq, rng):
        C = self._init_pp(X, X_sq, rng)
        labels = None
        for it in range(self.max_iter):
            D = _sq_dists(X, C, X_sq)
            new_labels = D.argmin(1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            C_new = np.zeros_like(C)
            np.add.at(C_new, labels, X)
            counts = np.bincount(labels, minlength=self.n_clusters)
            empty = counts == 0
            C_new[~empty] /= counts[~empty, None]
            if empty.any():  # re-seed an empty cluster at the point farthest from its centre
                far = D[np.arange(len(X)), labels].argsort()[::-1][: empty.sum()]
                C_new[empty] = X[far]
            shift = ((C_new - C) ** 2).sum()
            C = C_new
            if shift < self.tol:
                break
        D = _sq_dists(X, C, X_sq)
        labels = D.argmin(1)
        inertia = float(D[np.arange(len(X)), labels].sum())
        return C, labels, inertia, it + 1

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_sq = (X ** 2).sum(1)[:, None]
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, X_sq, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        return _sq_dists(np.asarray(X, dtype=np.float64), self.cluster_centers_).argmin(1)
