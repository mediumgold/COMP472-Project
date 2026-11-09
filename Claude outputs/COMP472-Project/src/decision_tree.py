"""
Decision Tree classifier (CART-style), implemented from scratch with NumPy.

How it learns
-------------
Start with all training samples in one node. Try every (feature, threshold)
question of the form "is feature_j <= t ?". Pick the question whose two child
nodes are the most "pure" (lowest Gini impurity or entropy). Repeat inside each
child until a stopping rule fires. A leaf predicts the class distribution of the
training samples that ended up there.

Stopping rules / hyperparameters
  max_depth          - maximum number of questions on any path (None = unlimited)
  min_samples_split  - a node needs at least this many samples to be split
  min_samples_leaf   - each child must keep at least this many samples
  criterion          - "gini"    : 1 - sum_c p_c^2
                       "entropy" : -sum_c p_c log p_c
  max_features       - how many randomly chosen features to try at each node
                       (None = all; Random Forest uses sqrt(d))

Speed trick: candidate thresholds
---------------------------------
Trying every distinct value of every feature is too slow in pure NumPy for
16,000 x 500 data. Instead each feature gets up to `n_bins - 1` candidate
thresholds taken at its training-set quantiles (e.g. 31 thresholds for
n_bins=32). Each value is replaced by the index of its bin (a small integer),
so the class counts on each side of every threshold for every feature can be
computed at once with one np.bincount + a cumulative sum. Histogram-based
libraries such as LightGBM use the same idea.
"""
import numpy as np


class Binner:
    """Per-feature quantile thresholds + conversion of values to bin indices."""

    def __init__(self, n_bins=32):
        assert 2 <= n_bins <= 256
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        Q = np.quantile(X, qs, axis=0).astype(np.float32)        # (n_bins-1, d)
        d = X.shape[1]
        self.edges_ = [np.unique(Q[:, j]) for j in range(d)]   # duplicates removed
        self.n_thresholds_ = np.array([len(e) for e in self.edges_])
        # thresholds_[j, b] = value t such that "bin <= b"  <=>  "x <= t"
        self.thresholds_ = np.full((d, self.n_bins), np.inf, dtype=np.float32)
        for j, e in enumerate(self.edges_):
            self.thresholds_[j, : len(e)] = e
        self.valid_ = np.arange(self.n_bins)[None, :] < self.n_thresholds_[:, None]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges_):
            # side="left": x == e[b] lands in bin b, so (bin <= b) <=> (x <= e[b])
            Xb[:, j] = np.searchsorted(e, X[:, j], side="left")
        return Xb


def _xlogx(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    m = a > 0
    out[m] = a[m] * np.log(a[m])
    return out


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features=None, n_bins=32, random_state=None):
        assert criterion in ("gini", "entropy")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state

    # ------------------------------------------------------------------ impurity
    def _impurity(self, counts):
        n = counts.sum()
        p = counts / n
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def _weighted_child_impurity(self, left, right, nL, nR, n):
        """
        Weighted impurity of the two children for every (feature, threshold) at once.
        left/right: class counts, shape (features, thresholds, classes).
        Returns  (nL * imp(left) + nR * imp(right)) / n , shape (features, thresholds).
        """
        nLs, nRs = np.maximum(nL, 1), np.maximum(nR, 1)
        if self.criterion == "gini":
            # nL * gini(left) = nL - sum_c left_c^2 / nL
            g = (nL - (left ** 2).sum(-1) / nLs) + (nR - (right ** 2).sum(-1) / nRs)
        else:
            # nL * entropy(left) = nL log nL - sum_c left_c log left_c
            g = (_xlogx(nL) - _xlogx(left).sum(-1)) + (_xlogx(nR) - _xlogx(right).sum(-1))
        return g / n

    # ------------------------------------------------------------ split search
    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(d, int(mf))

    def _best_split(self, Xb, idx, yb, parent_imp, rng):
        """Return (feature, bin, gain) of the best split, or None."""
        n, d = len(idx), Xb.shape[1]
        B = self.n_bins
        # Only the classes present in this node matter: relabel them 0..C-1.
        # Deep nodes often hold 2-3 classes, so this makes the histograms much smaller.
        _, yb = np.unique(yb, return_inverse=True)
        C = int(yb.max()) + 1
        k = self._n_features_to_try(d)
        if k >= d:
            # all features: evaluate in memory-friendly chunks, keep the best
            chunk = max(1, min(d, 4_000_000 // n, 2_000_000 // (B * C)))
            groups, stop_at_first_valid = [np.arange(i, min(d, i + chunk)) for i in range(0, d, chunk)], False
        else:
            # random subset of k features; if none of them can split this node
            # (e.g. all constant here), try the next k, like scikit-learn does
            perm = rng.permutation(d)
            groups, stop_at_first_valid = [perm[i:i + k] for i in range(0, d, k)], True

        best = None
        for feats in groups:
            sub = Xb[np.ix_(idx, feats)].astype(np.int64)         # (n, m) bin indices
            m = len(feats)
            code = ((np.arange(m) * B)[None, :] + sub) * C + yb[:, None]
            hist = np.bincount(code.ravel(), minlength=m * B * C).reshape(m, B, C)
            left = np.cumsum(hist, axis=1).astype(np.float64)    # counts with bin <= b
            total = left[:, -1:, :]
            right = total - left
            nL = left.sum(-1)
            nR = n - nL
            valid = ((nL >= self.min_samples_leaf) & (nR >= self.min_samples_leaf)
                     & self._valid_thr[feats])
            if not valid.any():
                continue
            child = self._weighted_child_impurity(left, right, nL, nR, n)
            child[~valid] = np.inf
            j, b = np.unravel_index(np.argmin(child), child.shape)
            gain = parent_imp - child[j, b]
            if best is None or gain > best[2]:
                best = (int(feats[j]), int(b), float(gain))
            if stop_at_first_valid:
                break
        if best is None or best[2] <= 1e-12:
            return None
        return best

    # ----------------------------------------------------------------- fitting
    def fit(self, X, y, n_classes=None):
        y = np.asarray(y, dtype=np.int64)
        binner = Binner(self.n_bins).fit(X)
        Xb = binner.transform(X)
        return self.fit_binned(Xb, y, binner, np.arange(len(y)), n_classes)

    def fit_binned(self, Xb, y, binner, sample_idx, n_classes=None):
        """Grow the tree on rows `sample_idx` of pre-binned data (Random Forest reuses this)."""
        self.n_classes_ = int(n_classes or (y.max() + 1))
        self.n_features_ = Xb.shape[1]
        self._valid_thr = binner.valid_
        thresholds = binner.thresholds_
        rng = np.random.default_rng(self.random_state)
        max_depth = np.inf if self.max_depth is None else self.max_depth

        feature, threshold, left_child, right_child, value, depth_of = [], [], [], [], [], []
        importances = np.zeros(self.n_features_)

        def new_node(counts, depth):
            feature.append(-1); threshold.append(np.nan)
            left_child.append(-1); right_child.append(-1)
            value.append(counts / counts.sum()); depth_of.append(depth)
            return len(feature) - 1

        root_counts = np.bincount(y[sample_idx], minlength=self.n_classes_).astype(np.float64)
        stack = [(new_node(root_counts, 0), sample_idx, 0)]
        n_total = len(sample_idx)
        while stack:
            node, idx, depth = stack.pop()
            n = len(idx)
            yb = y[idx]
            counts = np.bincount(yb, minlength=self.n_classes_)
            if (depth >= max_depth or n < self.min_samples_split
                    or n < 2 * self.min_samples_leaf or counts.max() == n):
                continue                                         # stays a leaf
            parent_imp = self._impurity(counts)
            split = self._best_split(Xb, idx, yb, parent_imp, rng)
            if split is None:
                continue
            f, b, gain = split
            go_left = Xb[idx, f] <= b
            idx_l, idx_r = idx[go_left], idx[~go_left]
            importances[f] += gain * n / n_total
            feature[node] = f
            threshold[node] = thresholds[f, b]
            cl = np.bincount(y[idx_l], minlength=self.n_classes_).astype(np.float64)
            cr = np.bincount(y[idx_r], minlength=self.n_classes_).astype(np.float64)
            l_id, r_id = new_node(cl, depth + 1), new_node(cr, depth + 1)
            left_child[node], right_child[node] = l_id, r_id
            stack.append((r_id, idx_r, depth + 1))
            stack.append((l_id, idx_l, depth + 1))

        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold, dtype=np.float32)
        self.left_ = np.array(left_child, dtype=np.int64)
        self.right_ = np.array(right_child, dtype=np.int64)
        self.value_ = np.array(value, dtype=np.float32)
        self.node_depth_ = np.array(depth_of, dtype=np.int64)
        s = importances.sum()
        self.feature_importances_ = importances / s if s > 0 else importances
        return self

    # -------------------------------------------------------------- prediction
    @property
    def n_leaves(self):
        return int((self.feature_ < 0).sum())

    @property
    def depth(self):
        return int(self.node_depth_.max())

    def apply(self, X, max_depth=None):
        """Index of the node each sample ends in (optionally stop early at max_depth)."""
        X = np.asarray(X, dtype=np.float32)
        node = np.zeros(len(X), dtype=np.int64)
        active = np.arange(len(X))
        depth = 0
        while active.size and (max_depth is None or depth < max_depth):
            f = self.feature_[node[active]]
            active = active[f >= 0]                              # drop samples already at a leaf
            if not active.size:
                break
            nd = node[active]
            go_left = X[active, self.feature_[nd]] <= self.threshold_[nd]
            node[active] = np.where(go_left, self.left_[nd], self.right_[nd])
            depth += 1
        return node

    def predict_proba(self, X, max_depth=None):
        return self.value_[self.apply(X, max_depth)]

    def predict(self, X, max_depth=None):
        return self.predict_proba(X, max_depth).argmax(axis=1)
