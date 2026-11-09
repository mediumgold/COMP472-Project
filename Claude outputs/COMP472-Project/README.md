# COMP 472 Project: Visual Object Recognition & Sentiment Analysis

Team **COMP472_GroupXX** (replace XX with your group number) · Concordia University · Fall 2026

We compare four classifiers — **k-Nearest Neighbours, Decision Tree, Random Forest** (all three
implemented from scratch in NumPy) and a **feed-forward Neural Network** (PyTorch) — on two kinds of
pre-extracted features:

| Task | Training rows | Classes | Features |
|---|---|---|---|
| Image (visual object recognition) | 20,000 (400 / class) | 50 objects | 500-dim bag-of-visual-words histograms |
| Sentiment (TweetEval) | 12,000 (4,000 / class) | Negative / Neutral / Positive | 768-dim BERT mean-pooled embeddings |

plus an unsupervised **k-Means** clustering experiment (also from scratch).

**Main results, conclusions and talking points: see [`FINDINGS.md`](FINDINGS.md).**

---

## Quick start

```bash
pip install -r requirements.txt
# put the five competition CSVs in data/ (see "Data" below)
jupyter lab            # then run notebooks/01 … 08 in order
```

Or run everything headless, in order:

```bash
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

`submission.csv` (ready to upload to Kaggle) is written to the project root by notebook 08.
The notebooks are saved **with their outputs**, so you can read every result without re-running anything.

## Data

Put these files in `data/` (they are excluded from git by `.gitignore`: `sentiment_train.csv` is
over GitHub's 100 MB limit):

```
data/image_train.csv   data/image_test.csv
data/sentiment_train.csv   data/sentiment_test.csv
data/sample_submission.csv
```

The first load parses the CSVs and caches them as `data/cache_<task>.npz` for fast reloading.

## Project layout

```
src/                      our implementations (imported by every notebook)
  common.py               loading, the single validation split, preprocessing, seed, helpers
  knn.py                  k-NN (from scratch)
  decision_tree.py        Decision Tree (from scratch, CART with Gini / entropy)
  random_forest.py        Random Forest (from scratch, uses our Decision Tree)
  kmeans.py               k-Means (from scratch, k-means++ initialisation)
  neural_net.py           MLP + training loop (PyTorch)
  plotting.py             shared figure style
notebooks/
  01_explore_data         what the features look like, class balance, PCA views
  02_knn                  k, distance, normalisation, voting, distance-weight power
  03_decision_tree        depth, criterion, min samples leaf/split, threshold resolution
  04_random_forest        number of trees, max features, depth, min samples leaf; vs single tree
  05_neural_net           preprocessing, lr, size, activation, dropout, weight decay, batch, optimiser
  06_clustering           k-Means vs true labels, number of clusters, co-clustered classes
  07_error_analysis       results table, per-class metrics, confusions, shared mistakes, cost
  08_final_model_and_submission   choose final model on validation, retrain on all data, submission.csv
results/
  tables/                 every experiment's numbers as CSV (+ final_model.json)
  figures/                every plot as PNG (for slides / report)
  val_predictions/        each model's validation predictions (used by notebooks 07 and 08)
submission.csv
```

## Validation methodology

* **One fixed split for everything**: 80% train / 20% validation, *stratified* (each class keeps its
  proportion: 320/80 per image class, 3,200/800 per sentiment class), seed 42 —
  `src/common.py: stratified_split()`. Every model and every hyperparameter is compared on this same
  validation set.
* **No leakage**: preprocessing that learns anything (standardisation means/stds, PCA) is fitted on the
  training split only, then applied to validation (and, at the end, refitted on all training rows before
  predicting the test set).
* **Model selection on validation only.** The Kaggle leaderboard was never used to choose anything.
* **Final model**: the configuration chosen on validation is retrained on all 20,000 / 12,000 labelled rows
  (notebook 08).
* Caveat we are aware of: because many configurations were compared on the same validation set, the best
  validation score is slightly optimistic. Differences below ~1% should be treated as noise
  (4,000 image / 2,400 sentiment validation rows → standard error ≈ 0.8% / 1.0%).

## What is implemented by us vs. taken from libraries

| Component | Ours | Library |
|---|---|---|
| k-NN: distances, neighbour search, voting | ✔ NumPy | — |
| Decision Tree: split search (Gini/entropy), growth, prediction | ✔ NumPy | — |
| Random Forest: bootstrap, feature subsampling, voting, OOB | ✔ NumPy (+ `concurrent.futures` for parallel trees) | — |
| k-Means (k-means++, Lloyd iterations) | ✔ NumPy | — |
| Preprocessing (standardise, Hellinger, L2, PCA via SVD) | ✔ NumPy | — |
| Neural network architecture + training loop | ✔ | PyTorch `torch.nn`, `torch.optim`, `torch.utils.data` (allowed) |
| Metrics (F1, confusion matrix, NMI, ARI) | | scikit-learn `sklearn.metrics` (allowed: metrics) |
| Data loading / tables / plots | | pandas, matplotlib |

No scikit-learn model classes are used anywhere.

**Decision Tree design note.** To make a pure-NumPy tree fast enough on 16,000 × 500 data, each feature
is given up to 31 candidate thresholds at its training-set quantiles (`n_bins=32`), instead of every
distinct value. Notebook 03 shows that 32, 64 and 128 bins give practically the same accuracy.

## Reproducibility

* Python 3.13.18 · NumPy 2.5.3 · pandas 3.0.5 · matplotlib 3.11.2 · PyTorch 2.15.0 · scikit-learn 1.9.1
* Random seed **42** everywhere (`src/common.py: SEED`, `set_seed()`); each Random Forest tree and each
  network seed is derived from it.
* Random Forest gives identical results with 1 or many CPU cores (each tree's randomness comes from its own seed).
* PyTorch is limited to 1 CPU thread in notebooks 05 and 08: for these small networks it is the fastest
  setting, and it avoids run-to-run differences from multi-threaded arithmetic.
* No GPU needed. Approximate run times on a 2-core cloud machine (a typical laptop is faster):

  | Notebook | Time here | Notes |
  |---|---|---|
  | 01 explore | < 1 min | |
  | 02 k-NN | ~ 2 min | |
  | 03 Decision Tree | ~ 7 min | |
  | 04 Random Forest | ~ 40 min | uses all CPU cores (`N_JOBS = -1`); much faster on an 8-core laptop |
  | 05 Neural Network | ~ 30 min | ~70 training runs |
  | 06 clustering | ~ 8 min | |
  | 07 error analysis | < 1 min | reads saved validation predictions |
  | 08 final model + submission | ~ 1 min | |

  Notebooks 07 and 08 only need the files in `results/val_predictions/`, so they can be re-run on their own.

## Acknowledgements

This code was developed with the assistance of an AI tool (Claude, by Anthropic). The project rules
require AI assistance to be acknowledged in the report. Every team member should read `src/` and the
notebooks and be able to explain them: the TA may ask anyone about any part.
