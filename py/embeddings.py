"""
Embedding processing utilities for the EO Notebook Experiment App.
Handles embedding data manipulation, similarity computation, and dimensionality reduction.
Designed to run in Pyodide (no file I/O, no API calls — those happen in JS).
"""

import numpy as np
from collections import Counter

HELIX_ORDER = ["NUL", "DES", "INS", "SEG", "CON", "SYN", "ALT", "SUP", "REC"]


def build_operator_embeddings(verb_embeddings, verb_names, operator_assignments):
    """Compute operator centroid embeddings from verb embeddings and assignments.

    Args:
        verb_embeddings: numpy array (n_verbs, dim)
        verb_names: list of verb name strings
        operator_assignments: dict {verb_name: operator_name}

    Returns:
        numpy array (9, dim) — one centroid per operator in HELIX_ORDER
    """
    dim = verb_embeddings.shape[1]
    centroids = np.zeros((9, dim))

    name_to_idx = {name: i for i, name in enumerate(verb_names)}

    for oi, op in enumerate(HELIX_ORDER):
        indices = []
        for verb, assigned_op in operator_assignments.items():
            if assigned_op == op and verb in name_to_idx:
                indices.append(name_to_idx[verb])
        if indices:
            centroids[oi] = verb_embeddings[indices].mean(axis=0)

    return centroids


def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: numpy array (n, dim)

    Returns:
        numpy array (n, n) of cosine similarities
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normalized = embeddings / norms
    return normalized @ normalized.T


def assign_verbs_to_operators(verb_embeddings, operator_embeddings):
    """Assign each verb to its nearest operator by cosine similarity.

    Args:
        verb_embeddings: numpy array (n_verbs, dim)
        operator_embeddings: numpy array (9, dim)

    Returns:
        dict with assignments, similarities, distribution
    """
    v_norm = verb_embeddings / (np.linalg.norm(verb_embeddings, axis=1, keepdims=True) + 1e-10)
    o_norm = operator_embeddings / (np.linalg.norm(operator_embeddings, axis=1, keepdims=True) + 1e-10)
    sim = v_norm @ o_norm.T

    nearest_idx = np.argmax(sim, axis=1)
    nearest_sim = np.max(sim, axis=1)

    distribution = Counter()
    for idx in nearest_idx:
        distribution[HELIX_ORDER[idx]] += 1

    return {
        "nearest_idx": nearest_idx.tolist(),
        "nearest_sim": nearest_sim.tolist(),
        "sim_matrix": sim.tolist(),
        "distribution": {op: distribution.get(op, 0) for op in HELIX_ORDER},
    }


def compute_pca_projection(embeddings, n_components=2):
    """Compute PCA projection for visualization.

    Args:
        embeddings: numpy array (n, dim)
        n_components: number of components (default 2)

    Returns:
        dict with projected coordinates and variance explained
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embeddings)

    return {
        "coords": projected.tolist(),
        "variance_explained": [float(v) for v in pca.explained_variance_ratio_],
    }


def compute_umap_projection(embeddings, n_neighbors=30, min_dist=0.1):
    """Compute UMAP projection for visualization.

    Args:
        embeddings: numpy array (n, dim)

    Returns:
        dict with projected coordinates
    """
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, metric="cosine",
                       n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        projected = reducer.fit_transform(embeddings)
        return {"coords": projected.tolist(), "method": "umap"}
    except ImportError:
        # Fallback to PCA if UMAP not available
        return compute_pca_projection(embeddings)


def embeddings_from_flat_list(flat_list, dim):
    """Convert a flat list of floats to a numpy array of embeddings.

    Args:
        flat_list: list of floats (n_items * dim)
        dim: embedding dimension

    Returns:
        numpy array (n_items, dim)
    """
    arr = np.array(flat_list, dtype=np.float32)
    return arr.reshape(-1, dim)


def compute_embedding_stats(embeddings):
    """Compute statistics about an embedding matrix.

    Args:
        embeddings: numpy array (n, dim)

    Returns:
        dict with shape, norms, similarity stats
    """
    norms = np.linalg.norm(embeddings, axis=1)

    # Sample pairwise similarities (avoid O(n^2) for large n)
    n = len(embeddings)
    if n > 1000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, size=1000, replace=False)
        sample = embeddings[sample_idx]
    else:
        sample = embeddings

    sim = cosine_similarity_matrix(sample)
    upper_tri = sim[np.triu_indices(len(sample), k=1)]

    return {
        "shape": list(embeddings.shape),
        "n_items": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "pairwise_sim_mean": float(np.mean(upper_tri)),
        "pairwise_sim_std": float(np.std(upper_tri)),
        "pairwise_sim_min": float(np.min(upper_tri)),
        "pairwise_sim_max": float(np.max(upper_tri)),
    }
