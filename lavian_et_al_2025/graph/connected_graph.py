from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphResult:
    adjacency: sparse.csr_matrix
    edges: np.ndarray
    weights: np.ndarray
    correlations: Optional[np.ndarray]


def _normalize_traces(traces: np.ndarray, n_nodes: int) -> np.ndarray:
    if traces.ndim != 2:
        raise ValueError("traces must be 2D (time x nodes) or (nodes x time)")
    if traces.shape[0] != n_nodes and traces.shape[1] == n_nodes:
        traces = traces.T
    if traces.shape[0] != n_nodes:
        raise ValueError("traces shape does not match coords count")
    traces = traces.astype(np.float32, copy=False)
    mean = traces.mean(axis=1, keepdims=True)
    std = traces.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (traces - mean) / std


def _edge_correlations(z_traces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    t = z_traces.shape[1]
    norms = np.sqrt((z_traces**2).sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    z = z_traces / norms
    return np.einsum("ij,ij->i", z[edges[:, 0]], z[edges[:, 1]])


def _build_knn_edges(coords: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    edge_map: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j, d in zip(indices[i], distances[i]):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            prev = edge_map.get((a, b))
            if prev is None or d < prev:
                edge_map[(a, b)] = float(d)

    edges = np.array(list(edge_map.keys()), dtype=np.int64)
    weights = np.array([edge_map[tuple(e)] for e in edges], dtype=np.float32)
    return edges, weights


def _connect_components(
    coords: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    metric: str,
    max_iters: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    if n <= 1:
        return edges, weights

    for _ in range(max_iters):
        adj = sparse.csr_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
        )
        adj = adj + adj.T
        n_comp, labels = connected_components(adj, directed=False)
        if n_comp <= 1:
            return edges, weights

        nn = NearestNeighbors(n_neighbors=min(10, n), metric=metric)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        extra_edges = []
        extra_weights = []
        for comp in range(n_comp):
            comp_nodes = np.where(labels == comp)[0]
            best = (np.inf, None, None)
            for i in comp_nodes:
                for d, j in zip(distances[i], indices[i]):
                    if labels[j] != comp:
                        if d < best[0]:
                            best = (float(d), int(i), int(j))
                        break
            if best[1] is not None:
                a, b = (best[1], best[2]) if best[1] < best[2] else (best[2], best[1])
                extra_edges.append((a, b))
                extra_weights.append(best[0])

        if not extra_edges:
            break

        edges = np.vstack([edges, np.array(extra_edges, dtype=np.int64)])
        weights = np.concatenate([weights, np.array(extra_weights, dtype=np.float32)])

    return edges, weights


def build_connected_graph(
    coords: np.ndarray,
    traces: Optional[np.ndarray] = None,
    *,
    k: int = 10,
    metric: str = "euclidean",
    corr_threshold: Optional[float] = 0.3,
    ensure_connected: bool = True,
) -> GraphResult:
    """
    Build a connected, undirected graph from ROI coordinates with optional
    functional pruning using trace correlations.

    Parameters
    ----------
    coords
        Array of shape (n_nodes, 2|3) with spatial coordinates.
    traces
        Optional array of shape (n_nodes, n_timepoints) or (n_timepoints, n_nodes).
    k
        Number of spatial neighbors per node.
    metric
        Distance metric for kNN (default: euclidean).
    corr_threshold
        Minimum correlation to keep an edge when traces are provided.
        Set to None to skip correlation pruning.
    ensure_connected
        If True, adds nearest inter-component edges until connected.

    Returns
    -------
    GraphResult
        adjacency: CSR adjacency matrix
        edges: (m, 2) int array of edge indices
        weights: (m,) float array of edge weights (distance)
        correlations: (m,) float array or None
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 1:
        raise ValueError("coords must be a 2D array with at least one node")

    edges, weights = _build_knn_edges(coords, k=k, metric=metric)

    correlations = None
    if traces is not None:
        z_traces = _normalize_traces(np.asarray(traces), coords.shape[0])
        correlations = _edge_correlations(z_traces, edges)
        if corr_threshold is not None:
            keep = correlations >= corr_threshold
            edges = edges[keep]
            weights = weights[keep]
            correlations = correlations[keep]

    if ensure_connected:
        edges, weights = _connect_components(coords, edges, weights, metric=metric)
        if traces is not None and correlations is not None and len(edges) != len(correlations):
            z_traces = _normalize_traces(np.asarray(traces), coords.shape[0])
            correlations = _edge_correlations(z_traces, edges)

    n = coords.shape[0]
    adj = sparse.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    adj = adj + adj.T

    return GraphResult(adjacency=adj, edges=edges, weights=weights, correlations=correlations)
