"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 03: Classification Model — Training & Evaluation
=============================================================================
Domain    : CLASSIFICATION
Problem   : Predict vegetable price tier — Low / Medium / High / Very High
Algorithms:
  PRIMARY  → K-Nearest Neighbours (KNN)        [module algorithm]
  COMPARE  → Decision Tree Classifier
  COMPARE  → Gaussian Naive Bayes
  COMPARE  → Random Forest Classifier

Custom Classes:
  VegetableDataset     — wraps the dataset with utility methods
  PriceTierClassifier  — encapsulates training, evaluation, prediction
  ModelComparator      — runs comparative analysis across multiple models

Run     : python scripts/03_train_evaluate.py
Input   : data/clean/kalimati_model.csv
Output  : model/  charts/
=============================================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors         import KNeighborsClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.naive_bayes       import GaussianNB
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                       f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
os.makedirs("model",  exist_ok=True)
os.makedirs("charts", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 1: VegetableDataset
# Wraps the raw CSV with preprocessing utilities and typed properties
# ═════════════════════════════════════════════════════════════════════════════
class VegetableDataset:
    """
    Custom data structure wrapping the Kalimati Tarkari model dataset.
    Provides feature/label split, commodity lookup, and summary statistics.
    """

    FEATURE_COLS = [
        "CommodityEncoded",
        "Year", "Month", "DayOfYear", "WeekOfYear", "Quarter",
        "MonthSin", "MonthCos", "DaySin", "DayCos",
        "Lag7", "Lag30", "Roll7Mean", "PriceRange",
    ]
    TARGET_COL  = "PriceTier"
    TIER_ORDER  = ["Low", "Medium", "High", "Very High"]

    def __init__(self, csv_path: str):
        self._df = pd.read_csv(csv_path, low_memory=False)
        self._df["Date"] = pd.to_datetime(self._df["Date"])

        # Encode commodity
        self._commodity_encoder = LabelEncoder()
        self._df["CommodityEncoded"] = self._commodity_encoder.fit_transform(
            self._df["Commodity"])

        # Encode target label
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self.TIER_ORDER)

        print(f"  VegetableDataset loaded: {self._df.shape}  "
              f"| Commodities: {self._df['Commodity'].nunique()}")

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def X(self) -> pd.DataFrame:
        """Feature matrix (all rows)."""
        return self._df[self.FEATURE_COLS]

    @property
    def y(self) -> np.ndarray:
        """Encoded label vector."""
        return self._label_encoder.transform(self._df[self.TARGET_COL])

    @property
    def y_raw(self) -> pd.Series:
        """Raw string labels."""
        return self._df[self.TARGET_COL]

    @property
    def commodities(self) -> list:
        return self._commodity_encoder.classes_.tolist()

    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    @property
    def commodity_encoder(self) -> LabelEncoder:
        return self._commodity_encoder

    # ── Methods ─────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Return a summary dict of key dataset statistics."""
        return {
            "total_records":   len(self._df),
            "commodities":     self._df["Commodity"].nunique(),
            "features":        len(self.FEATURE_COLS),
            "date_range":      f"{self._df['Date'].min().date()} → {self._df['Date'].max().date()}",
            "class_distribution": self._df[self.TARGET_COL].value_counts().to_dict(),
            "price_stats":     self._df["Average"].describe().round(2).to_dict(),
        }

    def get_commodity_data(self, commodity: str) -> pd.DataFrame:
        """Return all rows for a single commodity."""
        return self._df[self._df["Commodity"] == commodity].copy()

    def class_balance(self) -> pd.Series:
        return self._df[self.TARGET_COL].value_counts()


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 2: PriceTierClassifier
# Wraps a sklearn estimator with training, evaluation, and prediction logic
# ═════════════════════════════════════════════════════════════════════════════
class PriceTierClassifier:
    """
    Custom classifier wrapper for vegetable price tier prediction.
    Handles scaling, training, cross-validation, and structured evaluation.
    """

    def __init__(self, name: str, estimator):
        self.name      = name
        self.estimator = estimator
        self.scaler    = StandardScaler()
        self._fitted   = False
        self.metrics_  = {}
        self.cv_scores_= {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PriceTierClassifier":
        """Scale features and train the estimator."""
        X_sc = self.scaler.fit_transform(X_train)
        self.estimator.fit(X_sc, y_train)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scale then predict class labels."""
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predicting.")
        return self.estimator.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (if supported by estimator)."""
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.scaler.transform(X))
        return None

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: list) -> dict:
        """Compute all classification metrics on the test set."""
        preds = self.predict(X_test)
        self.metrics_ = {
            "accuracy":  round(accuracy_score(y_test, preds),  4),
            "precision": round(precision_score(y_test, preds, average="weighted",
                                               zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds, average="weighted",
                                            zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, preds, average="weighted",
                                        zero_division=0), 4),
            "conf_matrix": confusion_matrix(y_test, preds).tolist(),
            "report":    classification_report(y_test, preds,
                                               target_names=label_names,
                                               zero_division=0),
        }
        self._preds = preds
        return self.metrics_

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """Run stratified k-fold cross-validation."""
        X_sc = self.scaler.fit_transform(X)   # refit on full set for CV
        skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.estimator, X_sc, y,
                                 cv=skf, scoring="f1_weighted")
        self.cv_scores_ = {
            "mean": round(scores.mean(), 4),
            "std":  round(scores.std(),  4),
            "all":  scores.round(4).tolist(),
        }
        return self.cv_scores_

    def save(self, directory: str):
        """Pickle the fitted estimator and scaler."""
        os.makedirs(directory, exist_ok=True)
        safe_name = self.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        with open(os.path.join(directory, f"{safe_name}_model.pkl"),  "wb") as f:
            pickle.dump(self.estimator, f)
        with open(os.path.join(directory, f"{safe_name}_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

    def summary_row(self) -> dict:
        """Return a one-row dict suitable for a results DataFrame."""
        return {
            "Model":     self.name,
            "Accuracy":  self.metrics_.get("accuracy",  "-"),
            "Precision": self.metrics_.get("precision", "-"),
            "Recall":    self.metrics_.get("recall",    "-"),
            "F1 Score":  self.metrics_.get("f1_score",  "-"),
            "CV F1 Mean":self.cv_scores_.get("mean",    "-"),
            "CV F1 Std": f"±{self.cv_scores_.get('std', '-')}",
        }


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 3: ModelComparator
# Orchestrates training of multiple classifiers and comparative analysis
# ═════════════════════════════════════════════════════════════════════════════
class ModelComparator:
    """
    Manages multiple PriceTierClassifier instances, runs all of them, and
    generates comparison tables and charts.
    """

    def __init__(self, dataset: VegetableDataset):
        self.dataset    = dataset
        self.classifiers: list[PriceTierClassifier] = []
        self.results_df: pd.DataFrame = None
        self._best: PriceTierClassifier = None

    def add(self, clf: PriceTierClassifier) -> "ModelComparator":
        self.classifiers.append(clf)
        return self

    def run(self, test_size: float = 0.20, random_state: int = 42) -> pd.DataFrame:
        """Train, evaluate, and cross-validate all registered classifiers."""
        X = self.dataset.X.values
        y = self.dataset.y
        le = self.dataset.label_encoder
        tier_names = le.classes_.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        self._X_test  = X_test
        self._y_test  = y_test
        self._tier_names = tier_names

        print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
        rows = []
        for clf in self.classifiers:
            print(f"\n  ── {clf.name} ──────────────────────────────────────")
            clf.fit(X_train, y_train)
            clf.evaluate(X_test, y_test, tier_names)
            clf.cross_validate(X_train, y_train)
            print(f"     Accuracy : {clf.metrics_['accuracy']:.4f}")
            print(f"     F1 Score : {clf.metrics_['f1_score']:.4f}")
            print(f"     CV F1    : {clf.cv_scores_['mean']:.4f} ± {clf.cv_scores_['std']:.4f}")
            print(clf.metrics_["report"])
            rows.append(clf.summary_row())

        self.results_df = pd.DataFrame(rows)
        best_name = self.results_df.loc[self.results_df["F1 Score"].idxmax(), "Model"]
        self._best = next(c for c in self.classifiers if c.name == best_name)
        print(f"\n  ✅ Best model: {best_name}  "
              f"(F1={self._best.metrics_['f1_score']:.4f})")
        return self.results_df

    @property
    def best(self) -> PriceTierClassifier:
        return self._best

    def save_all(self, directory: str):
        for clf in self.classifiers:
            clf.save(directory)

    def plot_comparison(self, save_path: str):
        """Bar chart comparing all classifiers across 4 metrics."""
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        colors = ["#2d6a4f", "#40916c", "#f77f00", "#e63946"]
        for ax, metric, color in zip(axes, metrics, colors):
            vals = self.results_df[metric].astype(float)
            bars = ax.bar(range(len(vals)), vals, color=color,
                          alpha=0.85, edgecolor="black", width=0.6)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(self.results_df["Model"],
                               rotation=20, ha="right", fontsize=8)
            ax.set_title(metric, fontsize=11, fontweight="bold")
            ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.08))
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        fig.suptitle("Model Comparison — Classification Metrics",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def plot_confusion_matrix(self, clf: PriceTierClassifier, save_path: str):
        """Heatmap confusion matrix for a single classifier."""
        cm = np.array(clf.metrics_["conf_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=self._tier_names,
                    yticklabels=self._tier_names,
                    linewidths=0.5)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"Confusion Matrix — {clf.name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def plot_cv_scores(self, save_path: str):
        """CV F1 scores with error bars."""
        means = [c.cv_scores_["mean"] for c in self.classifiers]
        stds  = [c.cv_scores_["std"]  for c in self.classifiers]
        names = [c.name               for c in self.classifiers]
        colors = ["#2d6a4f", "#f77f00", "#e63946", "#457b9d"]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=7,
                      color=colors[:len(names)], alpha=0.85, edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel("Weighted F1 Score", fontsize=12)
        ax.set_title("5-Fold Stratified Cross-Validation — F1 Score",
                     fontsize=13, fontweight="bold")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(max(0, min(means) - 0.05), 1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS 4: KNNHyperparameterSearch
# Manual grid search over k values for KNN
# ═════════════════════════════════════════════════════════════════════════════
class KNNHyperparameterSearch:
    """
    Custom k-value grid search for K-Nearest Neighbours classifier.
    Evaluates a range of k values and identifies the optimal setting.
    """

    def __init__(self, k_range: list = None):
        self.k_range = k_range or [1, 3, 5, 7, 9, 11, 15, 21]
        self.results_: list[dict] = []
        self.best_k_: int = None

    def search(self, X_train, y_train, X_val, y_val) -> int:
        """Evaluate each k and return the best value by F1 score."""
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_v_sc  = scaler.transform(X_val)

        print(f"\n  KNN Hyperparameter Search over k = {self.k_range}")
        for k in self.k_range:
            knn   = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr_sc, y_train)
            preds = knn.predict(X_v_sc)
            f1    = f1_score(y_val, preds, average="weighted", zero_division=0)
            acc   = accuracy_score(y_val, preds)
            self.results_.append({"k": k, "f1": round(f1, 4), "accuracy": round(acc, 4)})
            print(f"    k={k:2d}  F1={f1:.4f}  Accuracy={acc:.4f}")

        best = max(self.results_, key=lambda r: r["f1"])
        self.best_k_ = best["k"]
        print(f"\n  ✅ Best k = {self.best_k_}  (F1={best['f1']:.4f})")
        return self.best_k_

    def plot(self, save_path: str):
        """Line chart of F1 vs k."""
        df = pd.DataFrame(self.results_)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["k"], df["f1"], "o-", color="#2d6a4f", lw=2, ms=8, label="F1 Score")
        ax.plot(df["k"], df["accuracy"], "s--", color="#f77f00", lw=2, ms=7, label="Accuracy")
        best_row = df.loc[df["f1"].idxmax()]
        ax.axvline(best_row["k"], color="red", linestyle=":", lw=1.5,
                   label=f"Best k={int(best_row['k'])}")
        ax.set_xlabel("k (Number of Neighbours)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("KNN Hyperparameter Tuning — k vs F1 Score",
                     fontsize=13, fontweight="bold")
        ax.legend(); ax.set_xticks(df["k"])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: Run training pipeline
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  STW5000CEM — Step 3: Classification Model Training")
    print("=" * 65)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset ...")
    dataset = VegetableDataset("data/clean/kalimati_model.csv")
    summary = dataset.summary()
    print(f"  Summary: {json.dumps({k:v for k,v in summary.items() if k != 'price_stats'}, indent=4)}")

    # ── KNN Hyperparameter Search ─────────────────────────────────────────────
    print("\n[2/5] KNN Hyperparameter Search ...")
    X_all = dataset.X.values
    y_all = dataset.y
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)

    knn_search = KNNHyperparameterSearch(k_range=[1, 3, 5, 7, 9, 11, 15, 21])
    best_k = knn_search.search(X_tr, y_tr, X_val, y_val)
    knn_search.plot("charts/knn_hyperparameter_search.png")

    # ── Build Comparator ──────────────────────────────────────────────────────
    print("\n[3/5] Training all classifiers ...")
    comparator = ModelComparator(dataset)
    comparator.add(PriceTierClassifier(
        f"KNN (k={best_k})",
        KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="euclidean")
    ))
    comparator.add(PriceTierClassifier(
        "Decision Tree",
        DecisionTreeClassifier(max_depth=10, min_samples_split=20,
                               random_state=42, class_weight="balanced")
    ))
    comparator.add(PriceTierClassifier(
        "Naive Bayes",
        GaussianNB(var_smoothing=1e-9)
    ))
    comparator.add(PriceTierClassifier(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, max_depth=12,
                               random_state=42, class_weight="balanced", n_jobs=-1)
    ))

    results_df = comparator.run(test_size=0.20, random_state=42)

    # ── Save Charts ───────────────────────────────────────────────────────────
    print("\n[4/5] Generating charts ...")
    comparator.plot_comparison("charts/model_comparison.png")
    comparator.plot_cv_scores("charts/cv_scores.png")
    comparator.plot_confusion_matrix(comparator.best, "charts/confusion_matrix_best.png")

    # Confusion matrices for all classifiers
    for clf in comparator.classifiers:
        safe = clf.name.replace(" ", "_").replace("(","").replace(")","").replace("=","")
        comparator.plot_confusion_matrix(clf, f"charts/cm_{safe}.png")

    # Feature importance chart (Decision Tree)
    dt_clf = next(c for c in comparator.classifiers if "Decision" in c.name)
    feat_imp = pd.DataFrame({
        "Feature":    VegetableDataset.FEATURE_COLS,
        "Importance": dt_clf.estimator.feature_importances_
    }).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(feat_imp["Feature"], feat_imp["Importance"],
            color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance — Decision Tree", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("charts/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: feature_importance.png")

    # ── Save Model Artefacts ──────────────────────────────────────────────────
    print("\n[5/5] Saving model artefacts ...")
    comparator.save_all("model")

    # Save best model with canonical name for CLI/GUI
    best = comparator.best
    with open("model/best_model.pkl",  "wb") as f: pickle.dump(best.estimator, f)
    with open("model/best_scaler.pkl", "wb") as f: pickle.dump(best.scaler, f)
    with open("model/commodity_encoder.pkl", "wb") as f:
        pickle.dump(dataset.commodity_encoder, f)
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(dataset.label_encoder, f)

    # Save last known prices for lag computation
    df_raw = pd.read_csv("data/clean/kalimati_model.csv", low_memory=False)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    last_prices = {}
    for veg in dataset.commodities:
        sub = df_raw[df_raw["Commodity"] == veg].sort_values("Date").tail(30)
        last_prices[veg] = {
            "prices": sub["Average"].tolist(),
            "dates":  sub["Date"].dt.strftime("%Y-%m-%d").tolist(),
        }
    with open("model/last_prices.json", "w") as f:
        json.dump(last_prices, f, indent=2)

    meta = {
        "best_model":      best.name,
        "feature_cols":    VegetableDataset.FEATURE_COLS,
        "tier_order":      VegetableDataset.TIER_ORDER,
        "commodities":     dataset.commodities,
        "dataset_summary": summary,
        "all_results": [c.summary_row() for c in comparator.classifiers],
        "knn_best_k":      int(best_k),
        "knn_search":      knn_search.results_,
    }
    with open("model/meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    results_df.to_csv("model/metrics.csv", index=False)

    print("\n── Final Results ─────────────────────────────────────────────")
    print(results_df.to_string(index=False))
    print(f"\n  Best Model : {best.name}")
    print(f"  Accuracy   : {best.metrics_['accuracy']:.4f}")
    print(f"  F1 Score   : {best.metrics_['f1_score']:.4f}")
    print("\n  Training complete ✅")
