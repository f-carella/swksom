"""Minimal utility functions for ClusterSOM.

This is a lightweight subset of the project-level utils module, keeping only
functions that are actually used by ClusterSOM for clustering and BMU lookup.
It is intentionally small and avoids the plotting-heavy helpers from the larger
utils.py file.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import psutil
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

__all__ = [
    "estimate_batch_size",
    "hex_coords_axial",
    "hex_distance",
    "NumOptiClust",
    "KMeansEvaluation",
]


def estimate_batch_size(n_nodes: int, dtype=np.float64, max_mem_frac: float = 0.2) -> int:
    """Return a safe batch size for BMU computations.

    Parameters
    ----------
    n_nodes : int
        Number of SOM nodes.
    dtype : numpy dtype, default np.float64
        Data type used for the distance matrix.
    max_mem_frac : float, default 0.2
        Fraction of available RAM to use.
    """
    avail_mem = psutil.virtual_memory().available
    bytes_per_float = np.dtype(dtype).itemsize
    max_bytes = avail_mem * max_mem_frac
    batch_size = int(max_bytes // (n_nodes * bytes_per_float))
    return max(1, min(batch_size, 10000))


def hex_coords_axial(rows: int, cols: int) -> np.ndarray:
    """Return axial hex grid coordinates for a SOM lattice."""
    coords = []
    for r in range(rows):
        for q in range(cols):
            coords.append((q, r - (q // 2)))
    return np.asarray(coords, dtype=int)


def hex_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hexagonal distance between two sets of axial coordinates."""
    a = np.asarray(a)
    b = np.asarray(b)
    dq = np.abs(a[:, 0] - b[:, 0])
    dr = np.abs(a[:, 1] - b[:, 1])
    d_combined = np.abs((a[:, 0] + a[:, 1]) - (b[:, 0] + b[:, 1]))
    return np.max(np.stack([dq, dr, d_combined], axis=0), axis=0)


def KMeansEvaluation(data: np.ndarray, prediction: np.ndarray, score_type: str):
    """Return the requested clustering score for KMeans output."""
    score_type = score_type.lower()

    if score_type in {"ch", "calinski", "calinski-harabasz", "calinski_harabasz"}:
        return calinski_harabasz_score(data, prediction)

    if score_type in {"silhouette", "silhouette_score"}:
        return silhouette_score(data, prediction)

    raise ValueError(f"Unsupported score type: {score_type!r}")


def NumOptiClust(data: np.ndarray, method: str = "Kneedle", clr=None, folder=None, plot: bool = False):
    """Choose an approximate optimal number of clusters for KMeans.

    This lightweight version keeps the core strategy used by ClusterSOM:
    - Kneedle elbow selection on inertia, when method == 'Kneedle'
    - fallback to silhouette score otherwise
    """
    n_clust = np.arange(2, 10, 1)

    if method == "Kneedle":
        inertia = []
        ch_scores = []

        for clust in n_clust:
            kmeans_fit = KMeans(n_clusters=clust, random_state=42, init="k-means++").fit(data)
            inertia.append(kmeans_fit.inertia_)
            ch_scores.append(KMeansEvaluation(data, kmeans_fit.predict(data), "CH"))

        knee = KneeLocator(n_clust, inertia, curve="convex", direction="decreasing")
        nopticlust = knee.knee

        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = ax1.twinx()
                ax1.plot(n_clust, inertia, "black", label="Inertia")
                ax2.plot(n_clust, ch_scores, "blue", label="Calinski-Harabasz")
                ax1.axvline(x=nopticlust, color="red", linestyle="--", label="Knee")
                ax1.set_xlabel("Clusters")
                ax1.set_ylabel("Inertia")
                ax2.set_ylabel("Calinski-Harabasz")
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2)
                if folder is not None:
                    os.makedirs(folder, exist_ok=True)
                    plt.savefig(os.path.join(folder, "knee_plot.png"))
                plt.show()
            except ImportError:
                pass

        return int(nopticlust)

    best_k = 2
    best_score = -np.inf
    for clust in n_clust:
        kmeans_fit = KMeans(n_clusters=clust, random_state=42, init="k-means++").fit(data)
        score = KMeansEvaluation(data, kmeans_fit.predict(data), "Silhouette")
        if score > best_score:
            best_score = score
            best_k = clust
    return int(best_k)

