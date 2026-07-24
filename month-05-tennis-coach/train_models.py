"""
Module A, step 3: train and compare 3 tiers of model on the phase-aggregated
feature table, same escalation pattern as the NTSB project:

    logistic regression (floor baseline)
    -> random forest (solid default)
    -> XGBoost (the likely-best, genuinely new one)

Split is by subject (actor), not by clip — same leakage concern as before:
a subject's clips must not appear in both train and test. Each model is
evaluated across multiple random group-splits (not just one), same rigor
habit as NTSB's "5 splits x 5 seeds" table, since a single split can lie.

Usage:
    pip install scikit-learn xgboost
    python train_models.py --features features.parquet
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

NON_FEATURE_COLS = {"clip_id", "actor", "action", "seq", "n_frames"}


def load_features(path):
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["action"]
    groups = df["actor"]
    return X, y, groups, feature_cols


def make_models():
    """One dict entry per tier — easy to add a 4th later without touching the rest."""
    models = {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=0)),
        ]),
    }
    if HAS_XGB:
        models["xgboost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                eval_metric="mlogloss", random_state=0,
            )),
        ])
    else:
        print("xgboost not installed — skipping that tier. `pip install xgboost` to add it.")
    return models


def run_one_split(models, X, y_encoded, groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y_encoded, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "macro_f1": f1_score(y_test, preds, average="macro"),
        }
    return results, (X_test, y_test)


def multi_seed_eval(models, X, y_encoded, groups, n_seeds=5):
    """Same rigor pattern as NTSB — one split can mislead, several can't as easily."""
    all_results = {name: {"accuracy": [], "macro_f1": []} for name in models}
    last_test_set = None

    for seed in range(n_seeds):
        results, test_set = run_one_split(models, X, y_encoded, groups, seed)
        last_test_set = test_set
        for name, metrics in results.items():
            all_results[name]["accuracy"].append(metrics["accuracy"])
            all_results[name]["macro_f1"].append(metrics["macro_f1"])

    return all_results, last_test_set


def print_summary(all_results, label_names):
    print("\n=== Model comparison across", len(next(iter(all_results.values()))["accuracy"]),
          "group-splits ===")
    for name, metrics in all_results.items():
        acc = np.array(metrics["accuracy"])
        f1 = np.array(metrics["macro_f1"])
        print(f"{name:20s}  accuracy {acc.mean():.3f} +/- {acc.std():.3f}"
              f"   macro-F1 {f1.mean():.3f} +/- {f1.std():.3f}"
              f"   (range {acc.min():.3f}-{acc.max():.3f})")


def print_feature_importance(models, feature_cols, top_n=10):
    for name in ("random_forest", "xgboost"):
        pipe = models.get(name)
        if pipe is None:
            continue
        clf = pipe.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            continue
        order = np.argsort(importances)[::-1][:top_n]
        print(f"\nTop {top_n} features — {name}:")
        for i in order:
            print(f"  {feature_cols[i]:35s} {importances[i]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    X, y, groups, feature_cols = load_features(args.features)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"{len(X)} clips, {X.shape[1]} features, {len(encoder.classes_)} classes: "
          f"{list(encoder.classes_)}")
    print(f"{groups.nunique()} unique subjects.")

    models = make_models()
    all_results, last_test_set = multi_seed_eval(models, X, y_encoded, groups, args.n_seeds)
    print_summary(all_results, encoder.classes_)
    print_feature_importance(models, feature_cols)

    # Confusion matrix from the last split's random forest, as a concrete example
    # to look at (not just the summary numbers) — which strokes get confused with which.
    X_test, y_test = last_test_set
    rf = models["random_forest"]
    preds = rf.predict(X_test)
    print("\nConfusion matrix (random forest, last split) — rows=true, cols=predicted:")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds),
        index=encoder.classes_, columns=encoder.classes_,
    ))
    print("\nFull classification report (random forest, last split):")
    print(classification_report(y_test, preds, target_names=encoder.classes_))


if __name__ == "__main__":
    main()
