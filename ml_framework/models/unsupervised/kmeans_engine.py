from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


class KMeansEngine:
    def __init__(self, k_range: range = range(2, 9), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state

        self.audit_df: pd.DataFrame = pd.DataFrame()
        self.elbow_k: int | None = None
        self.best_silhouette_k: int | None = None
        self.selected_k: int | None = None
        self.model: KMeans | None = None

    def _detect_elbow_max_distance(self, ks: list[int], inertias: list[float]) -> int:
        """
        Formal Elbow Detection via Normalized Maximum Perpendicular Distance (Geometric Kneedle):
        Normalizes (k, inertia) to [0, 1] and finds the point farthest from the chord
        connecting (k_min, inertia_max) to (k_max, inertia_min).
        """
        if len(ks) <= 2:
            return ks[0]

        k_arr = np.array(ks, dtype=float)
        i_arr = np.array(inertias, dtype=float)

        x_norm = (k_arr - k_arr[0]) / max(k_arr[-1] - k_arr[0], 1e-12)
        y_norm = (i_arr - i_arr[-1]) / max(i_arr[0] - i_arr[-1], 1e-12)

        # Distance from (x_norm, y_norm) to line x + y - 1 = 0
        perp_distances = np.abs(x_norm + y_norm - 1.0) / np.sqrt(2.0)
        best_idx = int(np.argmax(perp_distances))
        return ks[best_idx]

    def run_audit(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.shape[1] == 0:
            raise ValueError("No valid features remain after pruning/preprocessing to run K-Means.")
        n_samples = len(X)
        valid_ks = [k for k in self.k_range if 2 <= k < n_samples]
        if not valid_ks:
            raise ValueError("Dataset does not have enough rows for clustering audit.")

        records = []
        for k in valid_ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)

            records.append({
                "k": k,
                "inertia": round(float(km.inertia_), 4),
                "silhouette": round(float(silhouette_score(X, labels)), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 4),
                "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            })

        self.audit_df = pd.DataFrame(records)
        self.elbow_k = self._detect_elbow_max_distance(
            self.audit_df["k"].tolist(),
            self.audit_df["inertia"].tolist(),
        )
        best_sil_idx = int(self.audit_df["silhouette"].idxmax())
        self.best_silhouette_k = int(self.audit_df.loc[best_sil_idx, "k"])
        self.selected_k = self.elbow_k
        return self.audit_df

    def fit_predict(self, X: pd.DataFrame, k_override: int | None = None) -> pd.Series:
        if self.audit_df.empty:
            self.run_audit(X)

        chosen_k = k_override if k_override is not None else self.selected_k
        if chosen_k is None:
            raise RuntimeError("Failed to determine optimal K.")

        self.selected_k = chosen_k
        self.model = KMeans(n_clusters=chosen_k, n_init=10, random_state=self.random_state)
        labels = self.model.fit_predict(X)
        return pd.Series(labels, index=X.index, name="cluster")

    def plot_audit_and_pca(self, X: pd.DataFrame, save_dir: Path | None = None) -> None:
        if self.model is None or self.selected_k is None:
            labels = self.fit_predict(X)
        else:
            labels = pd.Series(self.model.labels_, index=X.index)

        assert self.model is not None and self.selected_k is not None

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # --- Panel 1: Elbow Method (Inertia + Formal Elbow Marker) ---
        ks = self.audit_df["k"].tolist()
        inertias = self.audit_df["inertia"].tolist()
        axes[0].plot(ks, inertias, marker="o", linewidth=2, label="Inertia (WCSS)")
        axes[0].plot([ks[0], ks[-1]], [inertias[0], inertias[-1]], linestyle="--", color="gray", label="Chord")
        axes[0].axvline(self.selected_k, color="red", linestyle=":", linewidth=2, label=f"Selected K={self.selected_k}")
        axes[0].set_title("Elbow Method (Inertia Sweep)")
        axes[0].set_xlabel("Number of Clusters (K)")
        axes[0].set_ylabel("Inertia")
        axes[0].legend()

        # --- Panel 2: Individual Cluster Silhouette Plot ---
        sample_silhouettes = silhouette_samples(X, labels)
        avg_silhouette = float(silhouette_score(X, labels))
        y_lower = 10

        for cluster_id in range(self.selected_k):
            cluster_vals = np.sort(sample_silhouettes[labels == cluster_id])
            size = len(cluster_vals)
            y_upper = y_lower + size

            color = plt.cm.tab10(cluster_id % 10)
            axes[1].fill_betweenx(
                np.arange(y_lower, y_upper), 0, cluster_vals, facecolor=color, edgecolor=color, alpha=0.8
            )
            axes[1].text(-0.05, y_lower + 0.5 * size, f"C{cluster_id}")
            y_lower = y_upper + 10

        axes[1].axvline(avg_silhouette, color="red", linestyle="--", label=f"Mean = {avg_silhouette:.3f}")
        axes[1].set_title(f"Silhouette Analysis per Cluster (K={self.selected_k})")
        axes[1].set_xlabel("Silhouette Coefficient")
        axes[1].set_ylabel("Cluster Samples")
        axes[1].set_yticks([])
        axes[1].legend()

        # --- Panel 3: 2D PCA Projection of Observations & Centroids ---
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        centers_df = pd.DataFrame(self.model.cluster_centers_, columns=X.columns)
        centroids_pca = pca.transform(centers_df)

        scatter = axes[2].scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7, edgecolors="k", s=45
        )
        axes[2].scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="red",
            marker="X",
            s=220,
            edgecolors="black",
            linewidths=1.5,
            label="Centroids",
        )
        var_explained = pca.explained_variance_ratio_ * 100
        axes[2].set_title("2D PCA Cluster Projection")
        axes[2].set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
        axes[2].set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
        axes[2].legend()
        fig.colorbar(scatter, ax=axes[2], label="Cluster ID")

        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "kmeans_audit_pca.png", dpi=150)
            plt.close(fig)
        else:
            plt.show()