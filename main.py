"""
main.py: run all four classifiers on both datasets with one command.

It uses the same 80/20 validation split as the notebooks and each model's best
configuration (found by the experiments in notebooks 02-05), trains every model,
and prints the validation-accuracy table required by the project.

Usage (from the project folder, with the CSVs in data/):
    python main.py                        # all 4 models x both tasks
    python main.py --task sentiment       # one task only
    python main.py --models knn,nn        # some models only (knn, dt, rf, nn)
    python main.py --rf-trees 100         # quicker Random Forest (400 = the reported result)
    python main.py --submission           # also retrain the final models on ALL
                                          # training data and write submission.csv

The notebooks contain the full experiments (all hyperparameter sweeps, plots,
clustering, error analysis); this script only re-runs the final configurations.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.common import (N_CLASSES, SEED, DATA_DIR, PROJECT_ROOT, TABLES_DIR, Preprocessor,
                        accuracy, get_split, load_task, set_seed)
from src.knn import KNNClassifier
from src.decision_tree import DecisionTreeClassifier
from src.random_forest import RandomForestClassifier
from src.neural_net import train_mlp, predict_proba

torch.set_num_threads(1)   # small networks: one thread is fastest (see README)

# Best configuration of each model per task, chosen on the validation set.
BEST = {
    "image": {
        "knn": dict(preprocessing="hellinger+standardize", k=100, metric="cosine",
                    weights="distance", power=16),                                   # notebook 02
        "dt": dict(criterion="gini", max_depth=None, min_samples_leaf=50),           # notebook 03
        "rf": dict(max_features="sqrt", max_depth=None, min_samples_leaf=1),         # notebook 04
        "nn": dict(preprocessing="standardize", hidden=(1024, 512), activation="relu",
                   dropout=0.0, lr=1e-3, optimizer="adam", weight_decay=0.0,
                   batch_size=128, epochs=60),                                       # notebook 05
    },
    "sentiment": {
        "knn": dict(preprocessing="none", k=50, metric="cosine", weights="distance", power=2),
        "dt": dict(criterion="gini", max_depth=None, min_samples_leaf=50),
        "rf": dict(max_features="sqrt", max_depth=None, min_samples_leaf=1),
        "nn": dict(preprocessing="none", hidden=(512, 256), activation="relu",
                   dropout=0.5, lr=1e-3, optimizer="adam", weight_decay=0.0,
                   batch_size=128, epochs=80),
    },
}
MODEL_NAMES = {"knn": "k-NN", "dt": "Decision Tree", "rf": "Random Forest", "nn": "Neural Network"}
NN_PATIENCE = 15          # early stopping: stop after 15 epochs without validation improvement

# Final Kaggle models (chosen in notebook 08):
#   image     -> k-NN (config above), with all 20,000 training images as neighbours
#   sentiment -> average of 5 networks (seeds 42-46), trained on all 12,000 tweets for
#                9 epochs (median of the best epochs found on validation in notebook 05)
FINAL_NN_SEEDS = 5
FINAL_NN_EPOCHS = 9


def split_prep(cfg):
    cfg = dict(cfg)
    return cfg.pop("preprocessing", "none"), cfg


def run_model(model, task, Xtr, ytr, Xva, yva, rf_trees, n_jobs):
    """Train one model on the training split and predict the validation split."""
    prep, cfg = split_prep(BEST[task][model])
    n_classes = N_CLASSES[task]
    t0 = time.perf_counter()
    if model in ("knn", "nn"):
        pp = Preprocessor(prep).fit(Xtr)              # fitted on the training split only
        A, B = pp.transform(Xtr), pp.transform(Xva)
    if model == "knn":
        clf = KNNClassifier(n_classes=n_classes, **cfg).fit(A, ytr)
        pred = clf.predict(B)
    elif model == "dt":
        clf = DecisionTreeClassifier(**cfg).fit(Xtr, ytr, n_classes)
        pred = clf.predict(Xva)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=rf_trees, random_state=SEED, n_jobs=n_jobs,
                                     **cfg).fit(Xtr, ytr, n_classes)
        pred = clf.predict(Xva)
    elif model == "nn":
        net, hist, best_epoch = train_mlp(A, ytr, B, yva, n_classes=n_classes, seed=SEED,
                                          patience=NN_PATIENCE, **cfg)
        pred = predict_proba(net, B).argmax(1)
    return pred, time.perf_counter() - t0


def validation_table(tasks, models, rf_trees, n_jobs):
    rows = []
    for task in tasks:
        Xtr, ytr, Xva, yva = get_split(task)
        print(f"\n=== {task}: {len(ytr)} training rows, {len(yva)} validation rows, "
              f"{N_CLASSES[task]} classes ===")
        for m in models:
            set_seed()
            pred, secs = run_model(m, task, Xtr, ytr, Xva, yva, rf_trees, n_jobs)
            acc = accuracy(yva, pred)
            rows.append(dict(task=task, method=MODEL_NAMES[m], val_acc=acc, seconds=round(secs, 1)))
            print(f"  {MODEL_NAMES[m]:15s} validation accuracy {acc:.4f}   ({secs:.1f}s)")
    df = pd.DataFrame(rows)
    table = df.pivot(index="method", columns="task", values="val_acc")
    table = table.reindex([MODEL_NAMES[m] for m in models])
    table.columns = [f"{c} val acc" for c in table.columns]
    print("\n=== Validation accuracy ===")
    print(table.map(lambda v: f"{v:.1%}").to_string())
    df.to_csv(TABLES_DIR / "main_results.csv", index=False)
    print(f"\n(saved to {(TABLES_DIR / 'main_results.csv').relative_to(PROJECT_ROOT)})")
    return df


def make_submission():
    """Retrain the final models on ALL labelled rows and write submission.csv."""
    print("\n=== Building submission.csv (final models trained on all training data) ===")
    preds = []

    X, y, X_test, ids = load_task("image")
    prep, cfg = split_prep(BEST["image"]["knn"])
    pp = Preprocessor(prep).fit(X)
    knn = KNNClassifier(n_classes=50, **cfg).fit(pp.transform(X), y)
    preds.append(pd.DataFrame({"id": ids, "label": knn.predict(pp.transform(X_test))}))
    print("  image:     k-NN done")

    X, y, X_test, ids = load_task("sentiment")
    prep, cfg = split_prep(BEST["sentiment"]["nn"])
    cfg["epochs"] = FINAL_NN_EPOCHS
    pp = Preprocessor(prep).fit(X)
    A, T = pp.transform(X), pp.transform(X_test)
    probs = []
    for s in range(FINAL_NN_SEEDS):
        net, _, _ = train_mlp(A, y, n_classes=3, seed=SEED + s, **cfg)
        probs.append(predict_proba(net, T))
    preds.append(pd.DataFrame({"id": ids, "label": np.mean(probs, axis=0).argmax(1)}))
    print(f"  sentiment: {FINAL_NN_SEEDS}-network ensemble done")

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub = sample[["id"]].merge(pd.concat(preds, ignore_index=True), on="id", how="left",
                               validate="one_to_one")
    assert len(sub) == 10000 and sub["label"].notna().all()
    sub["label"] = sub["label"].astype(int)
    img = sub["id"].str.startswith("img_")
    assert sub.loc[img, "label"].between(0, 49).all()
    assert sub.loc[~img, "label"].between(0, 2).all()
    sub.to_csv(PROJECT_ROOT / "submission.csv", index=False)
    print("  saved submission.csv (10,000 rows, all checks passed)")


def main():
    ap = argparse.ArgumentParser(description="Run all classifiers on both COMP 472 datasets.")
    ap.add_argument("--task", choices=["image", "sentiment", "both"], default="both")
    ap.add_argument("--models", default="knn,dt,rf,nn",
                    help="comma-separated subset of: knn, dt, rf, nn")
    ap.add_argument("--rf-trees", type=int, default=400,
                    help="trees in the Random Forest (400 = reported result; 100 is ~4x faster)")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="CPU cores for the Random Forest (-1 = all)")
    ap.add_argument("--submission", action="store_true",
                    help="also retrain the final models on all data and write submission.csv")
    ap.add_argument("--submission-only", action="store_true",
                    help="skip the validation table, only write submission.csv")
    args = ap.parse_args()

    tasks = ["image", "sentiment"] if args.task == "both" else [args.task]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from knn, dt, rf, nn")

    set_seed()
    start = time.perf_counter()
    if not args.submission_only:
        validation_table(tasks, models, args.rf_trees, args.jobs)
    if args.submission or args.submission_only:
        make_submission()
    print(f"\nTotal time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
