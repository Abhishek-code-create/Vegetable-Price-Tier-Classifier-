"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 04: K-Means Clustering Analysis
=============================================================================
Domain    : CLUSTERING (second AI domain for bonus marks)
Algorithm : K-Means Clustering
Purpose   : Discover natural vegetable market segments based on pricing
            behaviour — groups vegetables with similar price dynamics.

Run     : python scripts/04_clustering.py
Input   : data/clean/kalimati_clean.csv
Output  : charts/cluster_*.png
          model/kmeans_model.pkl
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

from sklearn.cluster     import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics       import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)
os.makedirs("model",  exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASS: VegetableCluster
# Encapsulates K-Means clustering on vegetable pricing profiles
# ═════════════════════════════════════════════════════════════════════════════
class VegetableCluster:
    """
    Custom K-Means clustering solution for vegetable price profiling.
    Groups vegetables into market segments based on aggregated price statistics.
    """

    PROFILE_FEATURES = [
        "mean_price", "median_price", "std_price",
        "min_price",  "max_price",
        "monsoon_avg",   # Jun–Aug average
        "winter_avg",    # Dec–Feb average
        "post_monsoon",  # Sep–Nov average
        "price_trend",   # linear slope over time
    ]

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler     = StandardScaler()
        self.model_     = None
        self.profiles_  = None
        self.labels_    = None

    def build_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-commodity price statistics into a profile matrix.
        Each row = one vegetable, columns = pricing profile features.
        """
        records = []
        for commodity, group in df.groupby("Commodity"):
            g = group.sort_values("Date")
            # Seasonal averages
            monsoon     = g[g["Month"].isin([6, 7, 8])]["Average"].mean()
            winter      = g[g["Month"].isin([12, 1, 2])]["Average"].mean()
            post_mon    = g[g["Month"].isin([9, 10, 11])]["Average"].mean()
            # Linear price trend (slope per year)
            if len(g) > 10:
                x = (g["Date"] - g["Date"].min()).dt.days.values.astype(float)
                y = g["Average"].values.astype(float)
                slope = np.polyfit(x, y, 1)[0] * 365  # NPR per year
            else:
                slope = 0.0
            records.append({
                "Commodity":    commodity,
                "mean_price":   g["Average"].mean(),
                "median_price": g["Average"].median(),
                "std_price":    g["Average"].std(),
                "min_price":    g["Average"].min(),
                "max_price":    g["Average"].max(),
                "monsoon_avg":  monsoon if not np.isnan(monsoon)  else g["Average"].mean(),
                "winter_avg":   winter  if not np.isnan(winter)   else g["Average"].mean(),
                "post_monsoon": post_mon if not np.isnan(post_mon) else g["Average"].mean(),
                "price_trend":  slope,
            })
        self.profiles_ = pd.DataFrame(records).set_index("Commodity")
        return self.profiles_

    def find_optimal_k(self, k_range: range = range(2, 9)) -> int:
        """Elbow method + silhouette score to find optimal k."""
        X = self.scaler.fit_transform(self.profiles_[self.PROFILE_FEATURES])
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_))
            print(f"    k={k}  Inertia={km.inertia_:.1f}  Silhouette={sil_scores[-1]:.4f}")
        best_k = list(k_range)[np.argmax(sil_scores)]
        # Plot elbow
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(list(k_range), inertias, "o-", color="#2d6a4f", lw=2, ms=8)
        axes[0].set_xlabel("k (Number of Clusters)"); axes[0].set_ylabel("Inertia (WCSS)")
        axes[0].set_title("Elbow Method — Optimal k", fontsize=12, fontweight="bold")
        axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
        axes[0].legend()
        axes[1].plot(list(k_range), sil_scores, "s-", color="#f77f00", lw=2, ms=8)
        axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score vs k", fontsize=12, fontweight="bold")
        axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
        axes[1].legend()
        fig.suptitle("K-Means Optimal Cluster Selection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("charts/cluster_elbow.png", dpi=150, bbox_inches="tight")
        plt.close()
        return best_k

    def fit(self, k: int) -> "VegetableCluster":
        """Fit K-Means with chosen k."""
        X = self.scaler.fit_transform(self.profiles_[self.PROFILE_FEATURES])
        self.model_ = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.labels_ = self.model_.fit_predict(X)
        self.profiles_["Cluster"] = self.labels_
        return self

    def cluster_labels(self) -> dict:
        """Auto-name clusters by their mean price level."""
        cluster_means = self.profiles_.groupby("Cluster")["mean_price"].mean().sort_values()
        names = ["Budget", "Mid-Range", "Premium", "Luxury"][:len(cluster_means)]
        return {int(idx): name for idx, name in zip(cluster_means.index, names)}

    def plot_pca(self, save_path: str):
        """2D PCA scatter of cluster assignments."""
        X = self.scaler.transform(self.profiles_[self.PROFILE_FEATURES])
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        cmap = plt.cm.Set1
        cluster_names = self.cluster_labels()
        fig, ax = plt.subplots(figsize=(10, 7))
        for c in sorted(self.profiles_["Cluster"].unique()):
            mask = self.profiles_["Cluster"] == c
            label = cluster_names.get(c, f"Cluster {c}")
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=90, label=label, alpha=0.8)
            # Annotate top-3 vegetables in each cluster
            idxs = np.where(mask)[0]
            for i in idxs[:3]:
                ax.annotate(self.profiles_.index[i],
                            (coords[i, 0], coords[i, 1]),
                            fontsize=7, alpha=0.7,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
        ax.set_title("K-Means Clustering — Vegetable Market Segments (PCA)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def plot_cluster_profiles(self, save_path: str):
        """Radar/bar profiles showing mean feature values per cluster."""
        cluster_names = self.cluster_labels()
        cluster_stats = self.profiles_.groupby("Cluster")[
            ["mean_price", "std_price", "monsoon_avg", "winter_avg", "price_trend"]
        ].mean().round(2)
        cluster_stats.index = [cluster_names.get(i, f"C{i}") for i in cluster_stats.index]
        fig, ax = plt.subplots(figsize=(11, 6))
        cluster_stats[["mean_price", "monsoon_avg", "winter_avg"]].plot(
            kind="bar", ax=ax, width=0.7,
            color=["#2d6a4f", "#f77f00", "#457b9d"], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Cluster"); ax.set_ylabel("Average Price (NPR/Kg)")
        ax.set_title("K-Means Cluster Profiles — Seasonal Price Comparison",
                     fontsize=13, fontweight="bold")
        ax.legend(["Overall Mean", "Monsoon Avg", "Winter Avg"])
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved: {save_path}")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model_, "scaler": self.scaler,
                         "profiles": self.profiles_}, f)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  STW5000CEM — Step 4: K-Means Clustering")
    print("=" * 65)

    df = pd.read_csv("data/clean/kalimati_clean.csv", low_memory=False)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month

    vc = VegetableCluster()
    print("\n[1/3] Building commodity price profiles ...")
    profiles = vc.build_profiles(df)
    print(f"  Profiles shape: {profiles.shape}")
    print(profiles.head())

    print("\n[2/3] Finding optimal k ...")
    best_k = vc.find_optimal_k(range(2, 8))
    print(f"   Saved: charts/cluster_elbow.png")
    vc.fit(best_k)

    cluster_names = vc.cluster_labels()
    print(f"\n  Cluster names: {cluster_names}")
    print("\n  Cluster membership:")
    for c, name in cluster_names.items():
        members = profiles[profiles["Cluster"] == c].index.tolist()
        print(f"    {name} ({c}): {members}")

    # Metrics
    X_sc = vc.scaler.transform(profiles[VegetableCluster.PROFILE_FEATURES])
    sil  = silhouette_score(X_sc, vc.labels_)
    db   = davies_bouldin_score(X_sc, vc.labels_)
    print(f"\n  Silhouette Score   : {sil:.4f}  (higher=better, max=1.0)")
    print(f"  Davies-Bouldin Idx : {db:.4f}   (lower=better, min=0.0)")

    print("\n[3/3] Generating cluster charts ...")
    vc.plot_pca("charts/cluster_pca.png")
    vc.plot_cluster_profiles("charts/cluster_profiles.png")
    vc.save("model/kmeans_model.pkl")
    print("   Saved: model/kmeans_model.pkl")

    # Save cluster assignments
    result = profiles[["mean_price", "Cluster"]].copy()
    result["ClusterName"] = result["Cluster"].map(cluster_names)
    result.sort_values("Cluster").to_csv("model/cluster_assignments.csv")
    print("   Saved: model/cluster_assignments.csv")

    with open("model/cluster_meta.json", "w") as f:
        json.dump({
            "n_clusters": best_k,
            "cluster_names": {str(k): v for k, v in cluster_names.items()},
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
        }, f, indent=2)

    print("\n  Clustering complete ✅")
