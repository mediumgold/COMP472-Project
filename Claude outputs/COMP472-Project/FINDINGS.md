# Findings and talking points

Every number below comes from the notebooks in `notebooks/` (saved with outputs) and the CSVs in
`results/tables/`. Figures for slides are in `results/figures/`. "Validation" always means our fixed,
stratified 20% hold-out (4,000 images / 2,400 tweets). With that many rows, differences below ~1% are noise.

---

## 1. Headline results (validation accuracy, best configuration of each method)

| Method | Image (50 classes, chance 2%) | Sentiment (3 classes, chance 33%) | Best configuration |
|---|---|---|---|
| k-NN | **58.1%** | 59.3% | Image: Hellinger + standardise, cosine, k = 100, votes ∝ 1/distance¹⁶ · Sentiment: raw, cosine, k = 50, votes ∝ 1/distance² |
| Decision Tree | 11.9% | 49.0% | Gini, unlimited depth, min 50 samples per leaf (both tasks) |
| Random Forest | 36.6% | 60.3% | 400 trees, √d features per split, fully grown trees |
| Neural Network | 39.1% | **68.3%** | Image: 500→1024→512→50, ReLU, Adam 1e-3, standardised input · Sentiment: 768→512→256→3, dropout 0.5 |

Table: `results/tables/summary_validation.csv` · Figure: `07_summary_accuracy.png`.
Neural-network scores are the best run of the search; averaged over 5 random seeds they are 38.8% ± 0.6%
(image) and 68.0% ± 0.3% (sentiment).

**Submitted to Kaggle (`submission.csv`):**
* **Image → k-NN** (config above), using all 20,000 training images as neighbours. Validation 58.1%.
* **Sentiment → average of 5 neural networks** (different seeds), retrained on all 12,000 tweets for 9 epochs. Validation 68.0%.
* If the test set behaves like validation: overall score ≈ (58% + 68%) / 2 ≈ **63%**. See §9 for two caveats.

---

## 2. The two datasets are fundamentally different (notebook 01)

**Image features are bag-of-visual-words histograms.** Every vector is non-negative, ~85% zeros, has length
exactly 1, and its non-zero values are whole-number multiples of each other, i.e. scaled counts of 500
"visual words" (local image patches). Consequences:
* Euclidean and cosine distance pick the *same* neighbours on the raw data (unit vectors: ‖a−b‖² = 2 − 2cos).
* The Hellinger transform (divide by total count, then square root), the classic fix for histograms,
  helps k-NN (+4.4 points).
* A single feature ("is visual word 230 present?") is weak evidence, which is bad news for trees.

**Sentiment features are dense BERT embeddings** with a few dimensions on very different scales (one has mean −3.5).

**Neither shows its classes in a 2-D PCA view.** The strongest directions of variation are about something
else. Along the Negative→Positive direction, the two extremes separate partially and **Neutral sits in the
middle overlapping both**, predicting (correctly) that Neutral is the hardest class.

---

## 3. Algorithm by algorithm: what we varied, what happened, why

### k-NN (notebook 02)
* **k behaves in opposite ways on the two tasks** (figure `02_knn_k_curves.png`).
  * Images: best at small k; accuracy *falls* as k grows (56.3% at k=1 → 26% at k=200 with equal votes). Large k underfits.
  * Sentiment: k = 1 is worst (48.7%, one noisy vote = overfitting); accuracy rises until k ≈ 50 (59%), then flattens.
* **Why:** for images, when the nearest training image is very close, 1-NN is right **88%** of the time; when
  it is far, only **20%** (`knn_accuracy_by_nn_distance.csv`). Each image class is made of many small, tight
  groups of near-relatives (plausibly several photos of the same object). One close neighbour finds the right
  group; more equal votes bring in other groups and other classes. For sentiment, closeness barely predicts
  correctness: no tight groups, noisy labels, so averaging many votes helps.
* **Distance weighting:** standard 1/distance votes help a little at large k (+1–1.5%). Our addition,
  1/distance^p with large p (p = 16), makes large k work for images (**58.1%**, best overall): the closest few
  neighbours decide, and the rest only break near-ties. Sentiment prefers mild weighting (p = 2).
* **Normalisation:** Hellinger helps images (51.9% → 56.3%). Standardising raw histograms with Euclidean distance
  *hurts* (43.0%): it inflates rare, noisy visual words. For sentiment all choices are within 0.7%.
* **Distance function:** identical on raw images (unit vectors). After standardising, cosine beats Euclidean by
  4–14 points on images; for sentiment the two are within ~0.6%.

### Decision Tree (notebook 03)
* **Sentiment shows textbook under/overfitting** (figure `03_dt_depth.png`). Depth 1–3 underfits (both accuracies
  low); validation peaks at depth 5 (48.5%); deeper trees reach 100% training accuracy while validation falls to 42.5%.
* **Images: overfits at every depth.** Training accuracy reaches 100% while validation crawls to ~11%. Each
  split tests one visual word, and the tree runs out of samples (320/class) long before combining enough weak evidence.
* **min_samples_leaf** is the most useful brake: sentiment 42.5% (leaf 1) → 49.0% (leaf 50) as training accuracy
  falls 100% → 62%. Images stay at 10.6–11.9% whatever we do.
* **Gini vs entropy:** no meaningful difference (48.5 vs 47.3%; 11.3 vs 10.5%).
* **Implementation note:** our tree tries up to 31 quantile thresholds per feature (`n_bins = 32`) for speed.
  32 bins is as accurate as 64 or 128 (image 11.9 / 11.8 / 11.7%, sentiment 49.0 / 48.9 / 47.8%), so the shortcut costs nothing.

### Random Forest (notebook 04)
* **Huge gain over a single tree:** +24.7 points on images (11.9% → 36.6%), +11.3 on sentiment (49.0% → 60.3%).
* **Number of trees** (figure `04_rf_n_trees.png`): sentiment levels off at ~60% after 100–150 trees; images are
  *still improving at 400 trees* (+1.3 points per extra 100). Fifty noisy class votes need many trees to average out.
* **Fully grown trees are best** (unlike a single tree): depth 5 → 13.9%, unlimited → 26.3% (images, 60 trees).
  Deep trees are low-bias/high-variance, and averaging removes the variance. This answers "does the Random Forest
  reduce overfitting?": its training accuracy is still 100%, but its error on unseen data is much lower.
* **max_features:** log2 (≈8) is too few. √d vs 10–25% are close. A 60-tree sweep favoured 10%, but at 400 trees √d
  won (36.6 vs 35.6%): small-forest differences were noise. We re-checked at full size before choosing.
* **OOB accuracy** (31.4% / 60.9%) is a free estimate from the ~37% of trees that didn't see each row. It is lower
  than validation on images because it effectively uses a ~150-tree forest.

### Neural Network (notebook 05)
* **Overfits fast** (figure `05_nn_baseline_curves.png`): 100% training accuracy by epoch ~10 (images) / ~30
  (sentiment), while validation loss starts rising after ~3 epochs (images) / ~10 epochs (sentiment). We keep the
  best-validation epoch (early stopping).
* **Images:** wider is better (no hidden layer 31.5% → 256-64 33.4% → 1024-512 **39.1%**). ReLU/GELU beat
  tanh/sigmoid by ~5 points. Learning rate 0.01 is too large (21%); 0.0001 works but needs ~10× more epochs.
  Dropout hurts. Weight decay 1e-3 helps slightly. 5-seed ensemble: 41.0%.
* **Sentiment:** every setting we tried lands between 67.2% and 68.3%, **including a network with no hidden layer
  (= logistic regression, 67.5%)**. Extra layers only add overfitting. Dropout 0.5 doesn't raise the peak but makes training
  much more forgiving (last-epoch 67.4% vs 64.0% without).
* **Why the network loses to k-NN on images:** it memorises the training set but must express each class as one
  smooth function. Recognising "close relative of training image #8,123" is what k-NN does naturally.

---

## 4. Cross-task comparison: does a method that wins on images also win on sentiment?

**No, the ranking flips.** Images: k-NN ≫ NN > RF ≫ DT. Sentiment: NN > RF ≈ k-NN ≫ DT.
* k-NN thrives on *local* structure (tight groups of near-relatives) → images.
* The network thrives when the signal is *spread thinly* across many dimensions and must be combined linearly → sentiment.
* Axis-aligned single-feature splits suit neither, so the single tree is last on both. Averaging trees (forest)
  rescues them substantially.

## 5. When does a more complex model pay off?

* **Sentiment: barely.** Logistic regression (67.5%) ≈ best deep network (68.3%), within noise. The BERT features
  seem to hold about 68% worth of linearly usable sentiment information. More capacity just memorises.
* **Images: yes, but only for the right kind of complexity.** Forest vs tree +25 points; wider networks +8 points.
  Yet the "simplest" method, k-NN, wins by ~20 points, because it matches the data's structure.
* **Ensembles** help unstable models (5 networks on images: 38.8% → 41.0%) but not stable ones (sentiment: 68.0 → 68.0),
  and averaging k-NN with the network *hurts* on images (51%), because the network's confident mistakes drown out
  k-NN's correct answers.

---

## 6. Clustering experiment: k-Means from scratch (notebook 06)

* **Sentiment does not cluster by sentiment.** With k = 3, every cluster contains plenty of all three labels:
  purity 0.42 (random clusters 0.34), NMI 0.02 (0 = unrelated). Two clusters lean slightly negative/positive. PCA
  to 50 dimensions first changes nothing. The supervised models find sentiment in weak directions that clustering ignores.
* **Image clusters follow the classes only weakly:** k = 50 → purity 0.11 (random 0.04), NMI 0.12. Purity and NMI
  keep rising up to k = 200 (0.17 / 0.20), consistent with "many small groups per class".
* **Frequently co-clustered classes:** 45 & 48, 2 & 3, 2 & 4, 12 & 25, 37 & 39 (`kmeans_image_coclustered_pairs.csv`).
  These overlap with the pairs the classifiers confuse (§7): the confusable classes are genuinely close in feature space.

## 7. Error analysis (notebook 07)

* **Neutral is the hardest sentiment class for every model** (F1 0.45–0.58 vs up to 0.75 for Negative). The
  network sends 24% of Neutral tweets to Negative and 23% to Positive; Negative→Positive errors are rarer (7%).
  k-NN almost refuses to predict Neutral (recall 35%). Negative is the easiest class. All models over-predict Negative a little.
* **Most-confused image pairs** (k-NN): 4→2 (14 of 80), 3→2 (11), 37↔39, 7↔8, 12↔14. Confusions are often
  one-directional (class 2 is a "hub"). The top five are also among k-Means' most co-clustered pairs; the top four are
  confused by the forest and the network as well.
* **Models agree on which image classes are hard** (per-class accuracy rank correlation 0.63–0.86). Hardest: 22, 27, 40, 9, 29.
  Easiest: 46, 48, 13, 5. (Only class numbers are available, not object names.)
* **Mistakes are shared:** 30.5% of validation images are missed by all four models (14.2% expected if independent);
  sentiment 15.0% (vs 2.6%).
* **Does the network fix simpler models' mistakes?** Sentiment: yes. It gets right ~40% of k-NN's and the forest's
  errors while breaking only 12–13% of their correct answers. Images: no. It fixes 18% of k-NN's errors but breaks 45% of its correct answers.
* **Confidence:** on sentiment, confidence is meaningful (≥ 0.95 → 93% correct). On images the network is
  over-confident (≥ 0.95 → only 63% correct), a symptom of memorisation.
* **Cost** (`compute_cost.csv`): k-NN trains instantly but every prediction scans all training rows. The forest is the
  most expensive to train (≈ 9 min for 400 image trees on 2 cores here). The network is moderate to train and
  near-free to apply.

---

## 8. How the final Kaggle model was selected

1. Every hyperparameter choice was made on the validation split (never the leaderboard).
2. Candidates compared on validation (`final_candidates.csv`): k-NN, Random Forest, 5-network ensemble, and
   probability averages of ensemble + k-NN (+ RF). Images: k-NN alone (58.1%) beat every combination. Sentiment: the
   5-network ensemble (68.0%) won.
3. We did **not** pick the single seed-42 network (68.3%) even though its number is higher. That run is the winner of
   the search on this same validation set, so its score is inflated by selection. Its 5-seed mean is 68.0%, and the
   ensemble is equally accurate and more stable.
4. The chosen models were retrained on all labelled rows (preprocessing re-fitted). For networks the number of epochs
   was fixed to the validation median (9), since no validation set remains for early stopping.

## 9. Caveats worth mentioning

* **Sentiment test predictions skew Negative** (46% of test tweets vs 38% on the balanced validation set). The test set
  probably has a different class mix from the balanced training data (TweetEval's original test split is not balanced),
  or slightly different tweets. We did not adjust anything based on the test set.
* **Image predictions are uneven across classes** (44–175 predicted per class, though the test set has exactly 100 of
  each): some classes are "hubs" in k-NN. *Optional, for the team to decide:* rescaling k-NN's class probabilities so
  every class receives an equal share (a few lines; uses only the stated fact of 100 per class) raised validation
  accuracy from 58.1% to 60.1% in a quick check. We left it out of the submission because it is a test-time
  adjustment, not one of the course algorithms. Ask the instructor if unsure.
* Best-of-search validation scores are slightly optimistic. Expect the leaderboard to land within a couple of points.

## 10. What we would do with more time

* More forest trees for images (still improving at 400), or a faster tree implementation.
* Learn a better image distance (e.g. a network trained so that same-class images are close) and use k-NN on top,
  combining the strengths of both.
* Cross-validation (5 folds) instead of one split, for tighter estimates when comparing close configurations.
* For sentiment: the features cap out near 68%; fine-tuning BERT itself would be the real lever but is not allowed here.

---

## 11. Questions the TA might ask (short answers)

* **Why stratified split?** Keeps every class's proportion identical in train and validation, so accuracies are comparable and no class is under-represented.
* **How do you avoid leakage?** Standardisation/PCA statistics are fitted on the training split only; the test set is never used for any choice.
* **Why is k = 1 so good on images but bad on sentiment?** Images come in tight groups of near-relatives (88% 1-NN accuracy when the neighbour is very close); tweets don't, and their labels are noisy, so many votes are needed.
* **Why does Euclidean = cosine on raw image features?** All vectors have length 1, and then ‖a−b‖² = 2 − 2cos(a,b), a monotonic function of cosine.
* **How does your tree choose a split?** For each feature and candidate threshold, count the classes on each side (one `np.bincount` + cumulative sum), compute the weighted Gini/entropy of the two children, keep the largest impurity decrease.
* **Why quantile thresholds instead of all values?** Speed in pure NumPy; 32 bins is as accurate as 128 (notebook 03).
* **What makes a Random Forest random?** Bootstrap rows per tree + a random subset of √d features at every node; prediction = average of the trees' class probabilities.
* **What is OOB accuracy?** Each training row scored only by trees that didn't see it: a free validation estimate.
* **Why does the forest have 100% training accuracy yet generalise better than the tree?** Each tree overfits differently; averaging cancels the differences (variance) while keeping their low bias.
* **What does early stopping do?** Keeps the weights from the epoch with the best validation accuracy, before the network starts memorising.
* **Why didn't more layers help sentiment?** A linear model already gets 67.5%; extra capacity only fits training noise.
* **What does the clustering tell you?** Sentiment isn't the dominant structure in the BERT features; image classes are only weakly clustered, and the classes k-Means mixes up are the ones the classifiers confuse.
