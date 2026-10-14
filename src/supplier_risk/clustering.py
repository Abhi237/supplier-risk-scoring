"""Seller archetypes: k-means on standardised reliability features, k chosen by silhouette (GMM BIC reported)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Name fragments for the dominant standardised centroid deviation (feature, sign) -> label
LABELS = {
    ("late_rate_shrunk", 1): "Chronic late deliverers",
    ("seller_handoff_late_rate", 1): "Slow dispatchers",
    ("mean_review_score", -1): "Service-quality risk",
    ("freight_to_price", 1): "Freight-heavy, low-ticket",
    ("cancel_rate", 1): "Cancellation-prone",
    ("log_gmv", 1): "High-volume core suppliers",
    ("median_delivery_days", 1): "Long-haul / remote lanes",
    ("median_delivery_days", -1): "Fast, reliable local",
    ("late_rate_shrunk", -1): "Reliable performers",
    ("log_gmv", -1): "Small occasional sellers",
    ("freight_to_price", -1): "High-ticket, low freight",
    ("mean_review_score", 1): "Highly rated",
    ("seller_handoff_late_rate", -1): "Fast dispatchers",
    ("cancel_rate", -1): "Low cancellation",
}


def select_k(X: np.ndarray, k_range, seed: int = 42) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=3).fit(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, km.labels_, sample_size=min(len(X), 5000),
                                                            random_state=seed),
                     "inertia": km.inertia_, "gmm_bic": gm.bic(X)})
    tab = pd.DataFrame(rows).round({"inertia": 4, "gmm_bic": 4})   # strip BLAS thread-order float noise
    return int(tab.loc[tab.silhouette.idxmax(), "k"]), tab


def name_clusters(z_centroids: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    used: set[str] = set()
    for c, row in z_centroids.iterrows():
        order = row.abs().sort_values(ascending=False).index
        for f in order:
            lab = LABELS.get((f, int(np.sign(row[f]) or 1)))
            if lab and lab not in used:
                names[c] = lab
                used.add(lab)
                break
        else:
            names[c] = f"Archetype {c}"
    return names


def fit_archetypes(df: pd.DataFrame, features: list[str], k_range, seed: int = 42):
    X = StandardScaler().fit_transform(df[features].to_numpy(float))
    k, tab = select_k(X, k_range, seed)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)
    # relabel clusters by size (largest = 0) for determinism
    order = pd.Series(km.labels_).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[x] for x in km.labels_])
    z = pd.DataFrame(X, columns=features).groupby(labels).mean()
    names = name_clusters(z)
    return labels, names, z, tab, X
