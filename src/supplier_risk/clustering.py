"""Vendor archetypes: k-means on standardised delivery-risk features, k chosen by silhouette (GMM BIC reported).

With a few dozen vendors these are INDICATIVE archetypes on a continuum, not separated segments.
"""
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
    ("mean_days_late_shrunk", 1): "Long-tail delays",
    ("freight_ratio", 1): "Freight-heavy, low-ticket",
    ("log_value", 1): "High-value core vendors",
    ("country_hhi", 1): "Single-country specialists",
    ("air_share", 1): "Air-dependent",
    ("air_share", -1): "Surface / truck-based",
    ("late_rate_shrunk", -1): "Reliable performers",
    ("mean_days_late_shrunk", -1): "Short delays",
    ("log_value", -1): "Small occasional vendors",
    ("freight_ratio", -1): "Low freight ratio",
    ("country_hhi", -1): "Multi-country suppliers",
}


def select_k(X: np.ndarray, k_range, seed: int = 42) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=3).fit(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_,
                     "gmm_bic": gm.bic(X)})
    tab = pd.DataFrame(rows).round({"silhouette": 6, "inertia": 4, "gmm_bic": 4})   # strip BLAS float noise
    return int(tab.loc[tab.silhouette.idxmax(), "k"]), tab


def name_clusters(z_centroids: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    used: set[str] = set()
    for c, row in z_centroids.iterrows():
        order = row.abs().sort_values(ascending=False, kind="mergesort").index
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
    df = df.sort_values("vendor", kind="mergesort")
    X = StandardScaler().fit_transform(df[features].to_numpy(float))
    k, tab = select_k(X, k_range, seed)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)
    # relabel clusters by size (largest = 0); ties broken by the lowest vendor index for determinism
    sizes = pd.Series(km.labels_).value_counts()
    first = pd.Series(range(len(km.labels_))).groupby(km.labels_).min()
    order = sorted(sizes.index, key=lambda c: (-sizes[c], first[c]))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[x] for x in km.labels_])
    z = pd.DataFrame(X, columns=features).groupby(labels).mean()
    names = name_clusters(z)
    return labels, names, z, tab, X, df.vendor.to_numpy()
