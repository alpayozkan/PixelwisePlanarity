"""
Plane parameter estimation from (depth, normal, plane_label).

Six algorithms exposed (cheapest → most robust):

    1. fit_planes_normal_average    — mean predicted normal + centroid offset
    2. fit_planes_least_squares     — point-on-plane + normal-consistency residuals
    3. fit_planes_svd               — orthogonal regression via SVD
    4. fit_planes_ransac            — RANSAC over points only
    5. fit_planes_ransac_normal     — RANSAC + normal-consistency AND-gate
    6. fit_planes_ransac_mestimator — RANSAC with Tukey biweight scoring

A unified dispatcher `compute_plane_params(method=...)` is also provided.

Inputs (all aligned at the same H × W resolution):
    depth        (H, W) float           depth in meters (or affine units;
                                        output `d` lives in the same units)
    normal       (H, W, 3) float        predicted unit normals in camera space
    plane_label  (H, W) int             0 = non-planar / background, 1+ = plane IDs

Camera handling:
    K=None  →  treat (u, v, depth) as 3D coordinates (matches pseudocode in
               the design doc; offset `d` is in pixel-space mixed units, NOT
               metric — useful for sanity checks only).
    K=(3,3) →  proper pinhole backprojection via `(u-cx)z/fx`, `(v-cy)z/fy`.
               Pass K when you want metric `d`.

Output:
    Dict[int, np.ndarray] mapping plane_id → array shape (4,) [a, b, c, d]
    with sqrt(a^2 + b^2 + c^2) = 1.

See also: `planamono.shared.plane_fitting.planefit.fit_planes_per_label_v1`,
which couples Open3D's RANSAC with an SVD refinement and is faster when you
already have a (N,3) point cloud.
"""

from typing import Dict, Optional, Tuple, Iterable

import numpy as np
from scipy.optimize import least_squares


__all__ = [
    "compute_plane_params",
    "fit_planes_normal_average",
    "fit_planes_least_squares",
    "fit_planes_svd",
    "fit_planes_ransac",
    "fit_planes_ransac_normal",
    "fit_planes_ransac_mestimator",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _gather_plane_data(
    depth: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray,
    K: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject masked-and-valid pixels; return (pts (N,3), n_pred (N,3))."""
    vs, us = np.nonzero(mask)
    z = depth[vs, us].astype(np.float64)
    keep = np.isfinite(z) & (z > 0)
    vs, us, z = vs[keep], us[keep], z[keep]

    if K is None:
        x = us.astype(np.float64)
        y = vs.astype(np.float64)
    else:
        K = np.asarray(K, dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)
    n_pred = np.asarray(normal)[vs, us].astype(np.float64)
    return pts, n_pred


def _normalize_plane(p: np.ndarray) -> Optional[np.ndarray]:
    nrm = float(np.linalg.norm(p[:3]))
    if nrm < 1e-12:
        return None
    return p / nrm


def _fit_plane_three(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Optional[np.ndarray]:
    n = np.cross(p2 - p1, p3 - p1)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-9:
        return None
    n = n / nrm
    d = -float(n @ p1)
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def _tukey_biweight(residuals: np.ndarray, c: float) -> np.ndarray:
    r = np.abs(residuals) / max(c, 1e-12)
    return np.where(r < 1.0, (1.0 - r * r) ** 2, 0.0)


def _iter_plane_masks(
    plane_label: np.ndarray,
    ignore_labels: Iterable[int],
    min_pixels: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    plane_label = np.asarray(plane_label)
    ignore = set(int(x) for x in ignore_labels)
    for pid in np.unique(plane_label):
        pid_int = int(pid)
        if pid_int in ignore:
            continue
        mask = plane_label == pid
        if int(mask.sum()) >= min_pixels:
            yield pid_int, mask


# --------------------------------------------------------------------------- #
# Algorithm 1: Direct Normal Average
# --------------------------------------------------------------------------- #

def fit_planes_normal_average(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, n_pred = _gather_plane_data(depth, normal, mask, K)
        if pts.shape[0] < 3:
            continue
        n = n_pred.mean(axis=0)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        n = n / nrm
        d = -float(n @ pts.mean(axis=0))
        out[pid] = np.array([n[0], n[1], n[2], d], dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Algorithm 2: Least Squares (joint point-distance + normal-consistency)
# --------------------------------------------------------------------------- #

def fit_planes_least_squares(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
    normal_weight: float = 1.0,
    max_nfev: int = 200,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, n_pred = _gather_plane_data(depth, normal, mask, K)
        if pts.shape[0] < 3:
            continue

        n0 = n_pred.mean(axis=0)
        nrm = float(np.linalg.norm(n0))
        if nrm < 1e-12:
            continue
        n0 = n0 / nrm
        d0 = -float(n0 @ pts.mean(axis=0))
        x0 = np.array([n0[0], n0[1], n0[2], d0], dtype=np.float64)

        N = pts.shape[0]

        def residuals(p):
            n = p[:3]
            d = p[3]
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                return np.full(2 * N, 1e6)
            nu = n / nn
            r_pts = pts @ nu + d / nn
            r_nrm = (1.0 - np.abs(n_pred @ nu)) * normal_weight
            return np.concatenate([r_pts, r_nrm])

        try:
            sol = least_squares(residuals, x0, method="lm", max_nfev=max_nfev).x
        except Exception:
            continue

        normalized = _normalize_plane(sol)
        if normalized is None:
            continue
        out[pid] = normalized
    return out


# --------------------------------------------------------------------------- #
# Algorithm 3: SVD (orthogonal regression)
# --------------------------------------------------------------------------- #

def fit_planes_svd(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
    orient_with_predicted_normal: bool = True,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, n_pred = _gather_plane_data(depth, normal, mask, K)
        if pts.shape[0] < 3:
            continue
        c = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        n = Vt[-1]
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        n = n / nrm
        # SVD normal direction is sign-ambiguous; align to mean predicted normal
        # so that downstream consumers see a consistent orientation.
        if orient_with_predicted_normal:
            ref = n_pred.mean(axis=0)
            if np.linalg.norm(ref) > 1e-12 and float(n @ ref) < 0:
                n = -n
        d = -float(n @ c)
        out[pid] = np.array([n[0], n[1], n[2], d], dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Algorithm 4: RANSAC (points only)
# --------------------------------------------------------------------------- #

def fit_planes_ransac(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
    residual_threshold: float = 0.05,
    max_trials: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(rng)
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, _ = _gather_plane_data(depth, normal, mask, K)
        N = pts.shape[0]
        if N < 3:
            continue

        best_p: Optional[np.ndarray] = None
        best_inliers = -1
        for _ in range(max_trials):
            idx = rng.choice(N, size=3, replace=False)
            p = _fit_plane_three(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if p is None:
                continue
            d_to_plane = np.abs(pts @ p[:3] + p[3])
            inliers = int(np.count_nonzero(d_to_plane < residual_threshold))
            if inliers > best_inliers:
                best_inliers = inliers
                best_p = p

        if best_p is not None:
            out[pid] = best_p
    return out


# --------------------------------------------------------------------------- #
# Algorithm 5: RANSAC + normal-consistency AND-gate
# --------------------------------------------------------------------------- #

def fit_planes_ransac_normal(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
    residual_threshold: float = 0.05,
    normal_threshold: float = 0.1,
    max_trials: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(rng)
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, n_pred = _gather_plane_data(depth, normal, mask, K)
        N = pts.shape[0]
        if N < 3:
            continue

        best_p: Optional[np.ndarray] = None
        best_score = -1
        for _ in range(max_trials):
            idx = rng.choice(N, size=3, replace=False)
            p = _fit_plane_three(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if p is None:
                continue
            d_to_plane = np.abs(pts @ p[:3] + p[3])
            point_in = d_to_plane < residual_threshold
            normal_in = (1.0 - np.abs(n_pred @ p[:3])) < normal_threshold
            score = int(np.count_nonzero(point_in & normal_in))
            if score > best_score:
                best_score = score
                best_p = p

        if best_p is not None:
            out[pid] = best_p
    return out


# --------------------------------------------------------------------------- #
# Algorithm 6: RANSAC with Tukey M-estimator (soft scoring)
# --------------------------------------------------------------------------- #

def fit_planes_ransac_mestimator(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    K: Optional[np.ndarray] = None,
    ignore_labels: Iterable[int] = (0,),
    min_pixels: int = 3,
    soft_threshold: float = 0.05,
    max_trials: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(rng)
    out: Dict[int, np.ndarray] = {}
    for pid, mask in _iter_plane_masks(plane_label, ignore_labels, min_pixels):
        pts, n_pred = _gather_plane_data(depth, normal, mask, K)
        N = pts.shape[0]
        if N < 3:
            continue

        best_p: Optional[np.ndarray] = None
        best_score = -np.inf
        for _ in range(max_trials):
            idx = rng.choice(N, size=3, replace=False)
            p = _fit_plane_three(pts[idx[0]], pts[idx[1]], pts[idx[2]])
            if p is None:
                continue
            d_to_plane = pts @ p[:3] + p[3]
            wp = _tukey_biweight(d_to_plane, soft_threshold)
            n_residual = 1.0 - np.abs(n_pred @ p[:3])
            wn = _tukey_biweight(n_residual, soft_threshold * 0.5)
            score = float(np.sum(wp * wn))
            if score > best_score:
                best_score = score
                best_p = p

        if best_p is not None:
            out[pid] = best_p
    return out


# --------------------------------------------------------------------------- #
# Unified dispatcher
# --------------------------------------------------------------------------- #

_METHODS = {
    "normal_average":    fit_planes_normal_average,
    "least_squares":     fit_planes_least_squares,
    "svd":               fit_planes_svd,
    "ransac":            fit_planes_ransac,
    "ransac_normal":     fit_planes_ransac_normal,
    "ransac_mestimator": fit_planes_ransac_mestimator,
}


def compute_plane_params(
    depth: np.ndarray,
    normal: np.ndarray,
    plane_label: np.ndarray,
    method: str = "ransac_normal",
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Dispatch to one of the six estimators by name.

    See module docstring for the list of methods and per-method kwargs.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {sorted(_METHODS)}"
        )
    return _METHODS[method](depth, normal, plane_label, **kwargs)
