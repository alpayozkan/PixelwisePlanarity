#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Planes-only pipeline with labels from per-face colors + legend CSV
RG + EM + Saturation + Policies + Recovery + Last-stage RG + Recursive Large Split
+ POST‑RANSAC over ALL remaining unlabeled faces (per label, no expansion)
+ FINAL PARAMETRIC MERGE BY (A,B,C,D)
+ normals.log writer (per plane + per label aggregates)

IMPORTANT (3‑POINT PLANES):
- All plane estimations are STRICT 3‑POINT fits (no IRLS/LSQ refinement).
- For any set of faces, we pick ONE triangle (largest-area face by default) and
  compute the plane from its 3 vertices.
- RANSAC returns the best 3‑point hypothesis WITHOUT refinement.

NEW (unit-aware scaling):
- --scale {original,metric,off}
- --scene_csv <metadata_scene.csv> and/or --meters_per_asset_unit <float>
- In 'original' mode, thresholds tuned for BASE_MPAU=0.02539999969303608 are
  scaled to the scene's meters_per_asset_unit to preserve physical lengths:
    S_dist = BASE_MPAU / MPAU_scene,  S_area = S_dist^2
- In 'metric' mode, thresholds specified in meters are converted to mesh units:
    S_dist = 1 / MPAU_scene,          S_area = S_dist^2
- In 'off' mode, no parameter scaling.
"""

import os, sys, json, argparse, csv, traceback
from typing import Tuple, Dict, List, Optional
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ------- tqdm (optional) -------
try:
    from tqdm import tqdm as _tqdm  # type: ignore
    def TQDM(iterable=None, total=None, desc=None, disable=False, **kw):
        return _tqdm(iterable=iterable, total=total, desc=desc, disable=disable, **kw)
    def PWRITE(msg: str):
        try: _tqdm.write(msg)
        except Exception: print(msg)
except Exception:
    class _DummyBar:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def close(self): pass
        def set_postfix(self, **kw): pass
    def TQDM(iterable=None, total=None, desc=None, disable=False, **kw):
        if iterable is not None: return iterable
        return _DummyBar()
    def PWRITE(msg: str): print(msg)

# ------- numba (optional) -------
NUMBA = False
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

# -------------------- small utilities / guards --------------------
def _ensure_face_index_vector(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        a = a.ravel()
    if a.dtype.kind not in "iu":
        a = a.astype(np.int64, copy=False)
    return np.unique(a).astype(np.int32, copy=False)

def safe_tri_from_face(F: np.ndarray, fid: int):
    row = np.asarray(F[int(fid)]).ravel()
    if row.size != 3:
        return None
    return int(row[0]), int(row[1]), int(row[2])

def _exclude_small_faces(faces: np.ndarray, FA: np.ndarray, thr: float) -> np.ndarray:
    faces = _ensure_face_index_vector(faces)
    if faces.size == 0: return faces
    return faces[FA[faces] >= float(thr)]

# -------------------- Hypersim legend + labels from PLY face colors --------------------
def _read_legend_csv(legend_csv_path: str):
    if not os.path.isfile(legend_csv_path):
        sys.exit(f"[ERR] legend CSV not found: {legend_csv_path}")
    color2info: Dict[int, Tuple[int, str]] = {}
    int2raw: Dict[int, str] = {}
    with open(legend_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        need = {"semantic_id", "semantic_name", "R", "G", "B"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"[ERR] legend CSV missing columns: {sorted(list(missing))}")
        for row in reader:
            try:
                sid = int(row["semantic_id"])
                name = str(row["semantic_name"])
                R = int(row["R"]); G = int(row["G"]); B = int(row["B"])
            except Exception:
                continue
            key = (np.uint32(R) << 16) | (np.uint32(G) << 8) | np.uint32(B)
            color2info[int(key)] = (sid, name)
            int2raw[sid] = name
    if not color2info:
        sys.exit("[ERR] legend CSV parsed but produced no entries.")
    return color2info, int2raw

def _labels_from_face_colors(mesh: trimesh.Trimesh, legend_csv_path: str) -> Tuple[np.ndarray, Dict[int,str]]:
    color2info, int2raw = _read_legend_csv(legend_csv_path)
    if not hasattr(mesh, "visual") or mesh.visual is None or mesh.visual.face_colors is None:
        sys.exit("[ERR] Mesh has no face colors. Load mesh_by_semantic.ply (per-face colors).]")
    FC = np.asarray(mesh.visual.face_colors)
    if FC.ndim != 2:
        sys.exit("[ERR] Unexpected face_colors array (ndim != 2).")
    if FC.shape[0] not in (mesh.faces.shape[0], mesh.vertices.shape[0]):
        sys.exit("[ERR] face_colors length mismatch; are colors per-face in the PLY?]")
    if FC.shape[0] == mesh.vertices.shape[0]:
        PWRITE("[WARN] Colors appear per-vertex; projecting to per-face via triangle vertex[0].")
        v0 = mesh.faces[:, 0]
        FC = FC[v0]
    if FC.shape[0] != mesh.faces.shape[0] or FC.shape[1] < 3:
        sys.exit("[ERR] Could not get per-face colors (need at least RGB).")
    RGB = FC[:, :3].astype(np.uint32)
    keys = (RGB[:, 0] << 16) | (RGB[:, 1] << 8) | RGB[:, 2]
    uk = np.unique(keys)
    lut = {}
    for k in uk:
        info = color2info.get(int(k))
        lut[int(k)] = info[0] if info is not None else -1
        if info is None:
            r = int((k >> 16) & 255); g = int((k >> 8) & 255); b = int(k & 255)
            PWRITE(f"[WARN] Color ({r},{g},{b}) not in legend; set label -1.")
    uk_sorted = np.sort(uk)
    lab_for_uk_sorted = np.array([lut[int(k)] for k in uk_sorted], dtype=np.int32)
    pos = np.searchsorted(uk_sorted, keys)
    labels_f = lab_for_uk_sorted[pos].astype(np.int32)
    return labels_f, int2raw

# -------------------- plane helpers (3‑point only) --------------------
def _normalize_nd(n: np.ndarray, d: float):
    """Unit-normalize n and co-scale D; flip so D>=0 (canonical form)."""
    n = np.asarray(n, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return np.array([0.0,0.0,1.0], dtype=np.float64), 0.0
    n = n / norm
    d = float(d) / norm
    if d < 0:
        n = -n; d = -d
    return n, d

def _plane_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    # n = normalize((b-a) x (c-a)); plane: n·x + d = 0; d = -n·a
    ab = b - a; ac = c - a
    n = np.cross(ab, ac)
    nrm = np.linalg.norm(n)
    if nrm < 1e-12:
        return None
    n = n / nrm
    d = -float(n @ a)
    n, d = _normalize_nd(n, d)
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)

def fit_plane_3pt_from_face(fid: int, F: np.ndarray, V: np.ndarray):
    tri = safe_tri_from_face(F, fid)
    if tri is None: return None
    a,b,c = tri
    return _plane_from_three_points(V[a], V[b], V[c])

def fit_plane_from_faces_3pt(face_ids: np.ndarray, F: np.ndarray, V: np.ndarray, FA: Optional[np.ndarray] = None):
    """Pick one triangle among face_ids (largest area if FA provided) and return its 3‑point plane."""
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return None
    if FA is not None and FA.size == F.shape[0]:
        fid = int(face_ids[np.argmax(FA[face_ids])])
    else:
        fid = int(face_ids[0])
    return fit_plane_3pt_from_face(fid, F, V)

# -------------------- sweep / gates --------------------
def sweep_inliers_label(n: np.ndarray, d: float,
                        faces_in_label: np.ndarray,
                        F: np.ndarray, V: np.ndarray, FN: np.ndarray, Cface: np.ndarray,
                        normal_deg: float, dist_m: float,
                        frac_vertices: float,
                        candidate_mask_label: np.ndarray,
                        gate_mode: str = "kof3") -> np.ndarray:
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    cand = faces_in_label[candidate_mask_label[faces_in_label]]
    if cand.size == 0: return np.empty(0, np.int32)
    cos_face = float(np.cos(np.deg2rad(normal_deg)))
    dots = FN[cand] @ n
    cand = cand[(dots > 0.0) & (np.abs(dots) >= cos_face)]
    if cand.size == 0: return np.empty(0, np.int32)
    if gate_mode == "none":
        return cand.astype(np.int32)
    if gate_mode == "centroid":
        td = np.abs(Cface[cand] @ n + d); return cand[td <= dist_m].astype(np.int32)
    tri_idx = F[cand].reshape(-1)
    dists = np.abs(V[tri_idx] @ n + d).reshape(-1, 3)
    need = int(np.ceil(3.0*frac_vertices))
    ok = (dists <= dist_m).sum(axis=1) >= need
    return cand[ok].astype(np.int32)

# ---- CSR adjacency (+ optional numba-accelerated local consensus) ----
def build_csr_from_adj_lists(adj_lists: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(adj_lists)
    lengths = np.fromiter((len(a) for a in adj_lists), count=n, dtype=np.int32)
    indptr = np.empty(n+1, np.int32); indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    indices = np.empty(indptr[-1], np.int32); pos = 0
    for a in adj_lists:
        m = len(a)
        if m:
            indices[pos:pos+m] = a
            pos += m
    return indptr, indices

if NUMBA:
    @njit(cache=True)
    def _filter_local_consensus_csr_numba(face_ids, indptr, indices, min_nbrs, nfaces):
        mask = np.zeros(nfaces, np.uint8)
        for i in range(face_ids.shape[0]): mask[face_ids[i]] = 1
        keep_count = 0
        for i in range(face_ids.shape[0]):
            f = face_ids[i]; start = indptr[f]; end = indptr[f+1]; cnt = 0
            for k in range(start, end):
                if mask[indices[k]] != 0: cnt += 1
            if cnt >= min_nbrs: keep_count += 1
        out = np.empty(keep_count, np.int32); j = 0
        for i in range(face_ids.shape[0]):
            f = face_ids[i]; start = indptr[f]; end = indptr[f+1]; cnt = 0
            for k in range(start, end):
                if mask[indices[k]] != 0: cnt += 1
            if cnt >= min_nbrs:
                out[j] = f; j += 1
        for i in range(face_ids.shape[0]): mask[face_ids[i]] = 0
        return out

def filter_local_consensus(face_ids: np.ndarray,
                           indptr: np.ndarray, indices: np.ndarray,
                           min_nbrs: int, nfaces: int) -> np.ndarray:
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return face_ids
    if NUMBA:
        return _filter_local_consensus_csr_numba(face_ids, indptr, indices, int(min_nbrs), int(nfaces))
    mask = np.zeros(nfaces, np.bool_)
    mask[face_ids] = True
    keep = []
    for f in face_ids:
        start, end = indptr[f], indptr[f+1]
        if np.count_nonzero(mask[indices[start:end]]) >= min_nbrs:
            keep.append(int(f))
    return np.array(keep, np.int32)

def split_by_plane_offset_clusters(face_ids: np.ndarray, Cface: np.ndarray, n: np.ndarray, d: float, gap=0.012):
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return []
    t = (Cface[face_ids] @ n) + d
    order = np.argsort(t)
    t_sorted = t[order]; idx_sorted = face_ids[order]
    comps, start = [], 0
    for i in range(1, t_sorted.size):
        if (t_sorted[i] - t_sorted[i-1]) > gap:
            comps.append(idx_sorted[start:i]); start = i
    comps.append(idx_sorted[start:])
    return comps

def eval_plane_quality(face_ids: np.ndarray, n: np.ndarray, d: float,
                       F: np.ndarray, V: np.ndarray, FN: np.ndarray, FA: np.ndarray,
                       p95_max: float, inlier_frac_min: float,
                       dist_thr: float,
                       normal_p95_deg_max: float,
                       thickness_max: float,
                       min_width_m: float,
                       fill_frac_min: float):
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return False, {}
    vidx = np.unique(F[face_ids].reshape(-1))
    P = V[vidx].astype(np.float64, copy=False)
    td = (P @ n) + d
    resid = np.abs(td)
    p95_res = float(np.percentile(resid, 95.0))
    inlier_frac = float(np.mean(resid <= dist_thr))
    thickness = float(td.max() - td.min())
    angs = np.degrees(np.arccos(np.clip(np.abs(FN[face_ids] @ n), -1.0, 1.0)))
    normal_p95 = float(np.percentile(angs, 95.0)) if angs.size>0 else 0.0
    def _basis(n: np.ndarray):
        a = np.array([1.0,0.0,0.0]) if abs(n[0]) < 0.9 else np.array([0.0,1.0,0.0])
        t1 = a - n*(a @ n); t1 /= (np.linalg.norm(t1)+1e-12)
        t2 = np.cross(n, t1); t2 /= (np.linalg.norm(t2)+1e-12)
        return t1, t2
    t1, t2 = _basis(n)
    X = P @ t1; Y = P @ t2
    w1 = float(X.max() - X.min()) if X.size else 0.0
    w2 = float(Y.max() - Y.min()) if Y.size else 0.0
    min_width = min(w1, w2)
    bbox_area = max(w1*w2, 1e-9)
    area = float(FA[face_ids].sum())
    fill = float(area / bbox_area)
    metrics = dict(p95=p95_res, inlier_frac=inlier_frac, thickness=thickness,
                   normal_p95=normal_p95, min_width=min_width, fill=fill, area=area)
    ok = (p95_res <= p95_max) and (inlier_frac >= inlier_frac_min) \
         and (thickness <= thickness_max) and (normal_p95 <= normal_p95_deg_max) \
         and (min_width >= min_width_m) and (fill >= fill_frac_min)
    return ok, metrics

def fit_and_quality(face_ids: np.ndarray, F: np.ndarray, V: np.ndarray, FN: np.ndarray, FA: np.ndarray,
                    p95_max: float, inlier_frac_min: float, dist_thr: float,
                    normal_p95_deg_max: float, thickness_max: float,
                    min_width_m: float, fill_frac_min: float,
                    irls_max_iters: int, irls_eps: float):
    # 3‑POINT plane from ONE triangle (largest area if FA provided)
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return False, None, {}
    pl = fit_plane_from_faces_3pt(face_ids, F, V, FA)
    if pl is None: return False, None, {}
    n, d = pl[:3], float(pl[3])
    ok, m = eval_plane_quality(face_ids, n, d, F, V, FN, FA,
                               p95_max, inlier_frac_min, dist_thr,
                               normal_p95_deg_max, thickness_max,
                               min_width_m, fill_frac_min)
    if not ok: return False, None, m
    return True, dict(n=n, d=d), m

# -------------------- RANSAC core (3‑point only, NO refine) --------------------
def ransac_plane_over_faces(face_ids: np.ndarray,
                            F: np.ndarray, V: np.ndarray,
                            FN: np.ndarray, Cface: np.ndarray,
                            dist_m: float, normal_deg: float,
                            max_iters: int, rng: np.random.Generator):
    face_ids = _ensure_face_index_vector(face_ids)
    verts = np.unique(F[face_ids].reshape(-1))
    if verts.size < 3: return None, np.empty(0, np.int32)
    cos_n = float(np.cos(np.deg2rad(normal_deg)))
    best_inliers = np.empty(0, np.int32); best_pl = None
    C = Cface[face_ids]; Nf = FN[face_ids]
    for _ in range(max(1, int(max_iters))):
        idx = rng.choice(verts, size=3, replace=False)
        pl = _plane_from_three_points(V[int(idx[0])], V[int(idx[1])], V[int(idx[2])])
        if pl is None: continue
        n, d = pl[:3], float(pl[3])
        td = np.abs(C @ n + d)
        na = np.abs(Nf @ n) >= cos_n
        inliers_mask = (td <= dist_m) & na
        inliers = face_ids[inliers_mask]
        if inliers.size > best_inliers.size:
            best_inliers = inliers; best_pl = pl
    if best_inliers.size == 0 or best_pl is None:
        return None, np.empty(0, np.int32)
    # NO refinement: return best 3‑point plane
    return best_pl, best_inliers.astype(np.int32)

# -------------------- components / adjacency helpers --------------------
def _components_in_subset(sub_idx: np.ndarray, adj_lists: List[np.ndarray], nfaces: int) -> List[np.ndarray]:
    sub_idx = _ensure_face_index_vector(sub_idx)
    if sub_idx.size == 0: return []
    mask = np.zeros(nfaces, np.bool_); mask[sub_idx] = True
    seen = np.zeros(nfaces, np.bool_)
    comps: List[np.ndarray] = []
    for s in sub_idx:
        s = int(s)
        if not mask[s] or seen[s]: continue
        stack = [s]; seen[s] = True; comp = [s]
        while stack:
            u = stack.pop()
            for v in adj_lists[u]:
                v = int(v)
                if mask[v] and not seen[v]:
                    seen[v] = True; stack.append(v); comp.append(v)
        comps.append(np.array(comp, np.int32))
    return comps

# -------------------- growth / merge / saturation --------------------
def grow_regions_in_label(faces_in_label: np.ndarray,
                          F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                          face_adj_lists: List[np.ndarray],
                          FN_gate: np.ndarray,
                          Cface: np.ndarray,
                          params: Dict,
                          visited_mask_label: np.ndarray,
                          gate_mode: str = "kof3") -> List[np.ndarray]:
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    in_label = np.zeros(F.shape[0], bool); in_label[faces_in_label] = True
    order = faces_in_label[np.argsort(-FA[faces_in_label])]
    regs = []
    cos_dihedral = float(np.cos(np.deg2rad(params['rg_dihedral_deg'])))
    cos_theta    = float(np.cos(np.deg2rad(params['rg_theta_deg'])))
    for seed in order:
        seed = int(seed)
        if not in_label[seed] or visited_mask_label[seed]:
            continue
        # initial plane from the seed face (3‑point)
        pl = fit_plane_3pt_from_face(seed, F, V)
        if pl is None:
            in_label[seed] = False; continue
        n, d = pl[:3], float(pl[3])
        stack = [seed]; in_label[seed] = False; reg = []
        cnt_since_refit = 0
        while stack:
            u = int(stack.pop())
            if visited_mask_label[u]: continue
            reg.append(u)
            cnt_since_refit += 1
            if cnt_since_refit >= params['rg_refit_every']:
                # re-anchor plane to the largest-area face currently in the region (3‑point)
                pl2 = fit_plane_from_faces_3pt(np.array(reg, np.int32), F, V, FA)
                if pl2 is not None: n, d = pl2[:3], float(pl2[3])
                cnt_since_refit = 0
            Nu = FN_gate[u]
            for v in face_adj_lists[u]:
                v = int(v)
                if not in_label[v] or visited_mask_label[v]: continue
                Nv = FN_gate[v]
                if abs(Nu @ Nv) < cos_dihedral: continue
                if abs(Nv @ n)  < cos_theta: continue
                if gate_mode == "centroid":
                    if abs(Cface[v] @ n + d) > params['rg_dist_m']: continue
                elif gate_mode == "kof3":
                    tri2 = safe_tri_from_face(F, v)
                    if tri2 is None: continue
                    a2,b2,c2 = tri2
                    dist3 = np.abs(V[[a2,b2,c2]] @ n + d)
                    if (dist3 <= params['rg_dist_m']).sum() < 2: continue
                in_label[v] = False; stack.append(v)
        if len(reg) >= params['min_faces_patch']:
            regs.append(np.array(reg, np.int32))
    return regs

def merge_within_label(patches_label, F, V, FA, Cface,
                       indptr, indices, nfaces,
                       merge_theta_deg, merge_dist_m,
                       sweep_normal_deg, sweep_dist_m, sweep_frac_vertices,
                       faces_in_label, FN, face_adj_lists, qual_params,
                       gate_mode: str,
                       irls_max_iters: int, irls_eps: float):
    if not patches_label: return patches_label
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    changed = True
    cos_merge = float(np.cos(np.deg2rad(merge_theta_deg)))
    while changed:
        changed = False; used = np.zeros(len(patches_label), bool); out=[]
        for i in range(len(patches_label)):
            if used[i]: continue
            pi = patches_label[i]; used[i]=True
            n1, d1 = pi['n'], float(pi['d'])
            for j in range(i+1, len(patches_label)):
                if used[j]: continue
                pj = patches_label[j]
                n2, d2 = pj['n'], float(pj['d'])
                if abs(n1 @ n2) < cos_merge: continue
                vA = np.unique(F[pi['faces']].reshape(-1)); vB = np.unique(F[pj['faces']].reshape(-1))
                dA2B = np.percentile(np.abs(V[vA] @ n2 + d2), 85) if vA.size>0 else np.inf
                dB2A = np.percentile(np.abs(V[vB] @ n1 + d1), 85) if vB.size>0 else np.inf
                if max(dA2B, dB2A) > merge_dist_m: continue
                faces_union = _ensure_face_index_vector(np.concatenate([pi['faces'], pj['faces']]))
                ok, plane, _ = fit_and_quality(faces_union, F, V, FN, FA,
                                               qual_params['p95_final_max'], qual_params['inlier_frac_min'],
                                               sweep_dist_m, qual_params['normal_p95_deg_max'],
                                               qual_params['thickness_max'], qual_params['min_width_m'],
                                               qual_params['fill_frac_min'],
                                               irls_max_iters, irls_eps)
                if not ok: continue
                nU, dU = plane['n'], float(plane['d'])
                cand_mask = np.ones(F.shape[0], bool)
                sweep = sweep_inliers_label(nU, dU, faces_in_label, F, V, FN, Cface,
                                            sweep_normal_deg, sweep_dist_m, sweep_frac_vertices, cand_mask,
                                            gate_mode=gate_mode)
                sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=2, nfaces=nfaces)
                faces_union = _ensure_face_index_vector(np.concatenate([faces_union, sweep]))
                ok2, plane2, _ = fit_and_quality(faces_union, F, V, FN, FA,
                                                 qual_params['p95_final_max'], qual_params['inlier_frac_min'],
                                                 sweep_dist_m, qual_params['normal_p95_deg_max'],
                                                 qual_params['thickness_max'], qual_params['min_width_m'],
                                                 qual_params['fill_frac_min'],
                                                 irls_max_iters, irls_eps)
                if ok2:
                    pi = dict(faces=faces_union, n=plane2['n'], d=float(plane2['d']),
                              area=float(FA[faces_union].sum()),
                              label_int=pi['label_int'], label_raw=pi['label_raw'], alg="Merge")
                    used[j]=True; changed=True
            out.append(pi)
        patches_label = out
    return patches_label

def saturate_within_label(patches_label, faces_in_label, F, V, FN, FA, face_adj_lists,
                          Cface,
                          indptr, indices, nfaces,
                          sat_normal_deg, sat_dist_m, sat_frac_vertices,
                          qual_params, rounds: int,
                          gate_mode: str,
                          irls_max_iters: int, irls_eps: float):
    if not patches_label: return patches_label
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    for _ in range(max(1, rounds)):
        patches_label.sort(key=lambda p: -p['area'])
        for p in patches_label:
            n, d = p['n'], float(p['d'])
            cand_mask = np.ones(F.shape[0], bool)
            sweep = sweep_inliers_label(n, d, faces_in_label, F, V, FN, Cface,
                                        sat_normal_deg, sat_dist_m, sat_frac_vertices, cand_mask,
                                        gate_mode=gate_mode)
            sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=1, nfaces=nfaces)
            union = _ensure_face_index_vector(np.concatenate([p['faces'], sweep]))
            ok, plane2, _ = fit_and_quality(union, F, V, FN, FA,
                                            qual_params['p95_final_max'], qual_params['inlier_frac_min'],
                                            sat_dist_m, qual_params['normal_p95_deg_max'],
                                            qual_params['thickness_max'], qual_params['min_width_m'],
                                            qual_params['fill_frac_min'],
                                            irls_max_iters, irls_eps)
            if ok:
                p['faces'] = union; p['n'] = plane2['n']; p['d'] = float(plane2['d'])
                p['area'] = float(FA[union].sum())

        # resolve overlaps greedily
        face2pid = -np.ones(F.shape[0], np.int32)
        patches_label.sort(key=lambda p: -p['area'])
        for pid, p in enumerate(patches_label):
            f = p['faces']; un = f[face2pid[f] < 0]; face2pid[un] = pid
        new_list = []
        for pid, p in enumerate(patches_label):
            f = p['faces']; keep = f[face2pid[f] == pid]
            if keep.size == 0: continue
            p['faces'] = keep; p['area'] = float(FA[keep].sum())
            new_list.append(p)
        patches_label = new_list

        # light merge
        patches_label = merge_within_label(
            patches_label, F, V, FA, Cface,
            indptr, indices, nfaces,
            merge_theta_deg=10.0, merge_dist_m=max(0.02, 0.6*sat_dist_m),
            sweep_normal_deg=sat_normal_deg, sweep_dist_m=sat_dist_m,
            sweep_frac_vertices=sat_frac_vertices,
            faces_in_label=faces_in_label, FN=FN, face_adj_lists=face_adj_lists,
            qual_params=qual_params, gate_mode=gate_mode,
            irls_max_iters=irls_max_iters, irls_eps=irls_eps
        )
    return patches_label

# -------------------- tiers / label passes --------------------
def compute_face_tiers4(faces_in_label: np.ndarray, FA: np.ndarray,
                        small_thr: float, normal_hi: float, big_hi: float):
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    if faces_in_label.size == 0:
        z = np.empty(0, np.int32)
        return z, z, z, z
    areas = FA[faces_in_label]
    small_mask  = areas < float(small_thr)
    normal_mask = (areas >= float(small_thr)) & (areas < float(normal_hi))
    big_mask    = (areas >= float(normal_hi)) & (areas < float(big_hi))
    huge_mask   = areas >= float(big_hi)
    small = faces_in_label[small_mask]
    normal = faces_in_label[normal_mask]
    big = faces_in_label[big_mask]
    huge = faces_in_label[huge_mask]
    return (small.astype(np.int32), normal.astype(np.int32),
            big.astype(np.int32), huge.astype(np.int32))

def seed_huge_faces_as_planes(lbl: int,
                              huge_faces: np.ndarray,
                              F: np.ndarray, V: np.ndarray, FN: np.ndarray, FA: np.ndarray,
                              args, int2raw: Dict[int, str]) -> List[Dict]:
    huge_faces = _ensure_face_index_vector(huge_faces)
    if huge_faces.size == 0: return []
    label_raw = str(int2raw.get(int(lbl), 'unannotated'))
    patches: List[Dict] = []
    for f in huge_faces:
        pl = fit_plane_3pt_from_face(int(f), F, V)
        if pl is None: continue
        n, d = pl[:3], float(pl[3])
        faces = np.array([int(f)], np.int32)
        patches.append(dict(
            faces=faces, n=n, d=d, area=float(FA[faces].sum()),
            label_int=int(lbl), label_raw=label_raw, alg="HUGE-SEED"
        ))
    return patches

def extract_big_patches_in_label(lbl: int,
                                 big_faces: np.ndarray,
                                 F: np.ndarray, V: np.ndarray, FN: np.ndarray, FA: np.ndarray,
                                 face_adj_lists: List[np.ndarray], Cface: np.ndarray,
                                 args, int2raw: Dict[int, str]) -> List[Dict]:
    big_faces = _ensure_face_index_vector(big_faces)
    if big_faces.size == 0: return []
    nfaces_total = F.shape[0]
    comps = _components_in_subset(big_faces, face_adj_lists, nfaces_total)
    rng = np.random.default_rng(123)
    patches: List[Dict] = []
    label_raw = str(int2raw.get(int(lbl), 'unannotated'))

    q_p95 = args.big_p95_final_max
    q_inlier = args.big_inlier_frac_min
    q_norm = args.big_normal_p95_deg_max
    q_thick = args.big_thickness_max_mul * args.sweep_dist_m
    q_minw = args.big_min_width_m
    q_fill = args.big_fill_frac_min

    for comp in comps:
        leftover = _ensure_face_index_vector(comp)
        planes_made = 0
        while leftover.size >= max(1, args.big_min_faces_patch):
            if int(args.big_ransac_enable) != 1: break
            if planes_made >= int(args.big_ransac_max_planes_per_comp if hasattr(args, "big_ransac_max_planes_per_comp") else 16):
                break
            pl, inliers = ransac_plane_over_faces(
                leftover, F, V, FN, Cface,
                dist_m=args.big_ransac_dist_m,
                normal_deg=args.big_ransac_normal_deg,
                max_iters=args.big_ransac_max_iters,
                rng=rng
            )
            if pl is None or inliers.size < max(args.big_min_faces_patch,
                                                int(np.ceil(leftover.size * args.big_ransac_min_inlier_frac))):
                break
            n, d = pl[:3], float(pl[3])
            ok, _ = eval_plane_quality(inliers, n, d, F, V, FN, FA,
                                       q_p95, q_inlier, args.big_ransac_dist_m,
                                       q_norm, q_thick, q_minw, q_fill)
            area = float(FA[inliers].sum())
            if ok and area >= float(args.big_min_area_patch):
                patches.append(dict(
                    faces=_ensure_face_index_vector(inliers),
                    n=n, d=float(d), area=area,
                    label_int=int(lbl), label_raw=label_raw, alg="BIG-RANSAC"
                ))
                in_set = set(int(x) for x in inliers.tolist())
                mask = np.array([int(f) not in in_set for f in leftover], dtype=bool)
                leftover = leftover[mask]
                planes_made += 1
            else:
                break
    return patches

def process_one_label_normal_only(lbl: int,
                                  F: np.ndarray, V: np.ndarray, FA: np.ndarray, FN: np.ndarray, FN_gate: np.ndarray,
                                  face_adj_lists: List[np.ndarray], labels_f: np.ndarray, int2raw: Dict[int,str],
                                  Cface: np.ndarray, indptr: np.ndarray, indices: np.ndarray, nfaces: int,
                                  args, faces_override: Optional[np.ndarray] = None) -> List[Dict]:
    faces_in_label = faces_override if faces_override is not None else np.where(labels_f==int(lbl))[0]
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    if faces_in_label.size < max(20, args.min_faces_patch//2): return []
    label_raw = str(int2raw.get(int(lbl), 'unannotated'))

    regs = grow_regions_in_label(
        faces_in_label, F, V, FA,
        face_adj_lists=face_adj_lists, FN_gate=FN_gate, Cface=Cface,
        params=dict(rg_dihedral_deg=args.rg_dihedral_deg,
                    rg_theta_deg=args.rg_theta_deg,
                    rg_dist_m=args.rg_dist_m,
                    rg_refit_every=args.rg_refit_every,
                    min_faces_patch=args.min_faces_patch),
        visited_mask_label=np.zeros(F.shape[0], bool),
        gate_mode=args.rg_gate_mode
    )

    areas = [FA[r].sum() for r in regs]
    order = np.argsort(-np.asarray(areas)) if areas else np.array([], dtype=int)
    regs_sorted = [regs[i] for i in order] if areas else []

    label_patches = []
    taken_label = np.zeros(F.shape[0], bool); taken_label[labels_f!=int(lbl)] = True
    for reg in regs_sorted:
        reg = _ensure_face_index_vector(reg)
        if taken_label[reg].all(): continue

        ok, plane, _ = fit_and_quality(reg, F, V, FN, FA,
                                       args.p95_final_max, args.inlier_frac_min, args.dist_thr,
                                       args.normal_p95_deg_max,
                                       args.thickness_max_mul*args.sweep_dist_m,
                                       args.min_width_m, args.fill_frac_min,
                                       args.irls_max_iters, args.irls_eps)
        if not ok: continue
        n, d = plane['n'], float(plane['d'])

        prev_sz = 0; curr = reg.copy()
        for _ in range(max(1, args.em_max_iters)):
            cand_mask = ~taken_label
            sweep = sweep_inliers_label(n, d, faces_in_label, F, V, FN, Cface,
                                        args.sweep_normal_deg, args.sweep_dist_m,
                                        args.sweep_frac_vertices, cand_mask,
                                        gate_mode=args.gate_mode)
            sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=2, nfaces=nfaces)
            union = _ensure_face_index_vector(np.concatenate([curr, sweep]))
            clusters = split_by_plane_offset_clusters(union, Cface, n, d, gap=max(0.010, 0.5*args.sweep_dist_m))
            if clusters:
                seed_set = set(int(x) for x in reg.tolist())
                best, best_ov = clusters[0], -1
                for cl in clusters:
                    ov = len(seed_set.intersection(set(int(x) for x in cl.tolist())))
                    if ov > best_ov: best, best_ov = cl, ov
                union = best
            ok2, plane2, _ = fit_and_quality(union, F, V, FN, FA,
                                             args.p95_final_max, args.inlier_frac_min, args.dist_thr,
                                             args.normal_p95_deg_max,
                                             args.thickness_max_mul*args.sweep_dist_m,
                                             args.min_width_m, args.fill_frac_min,
                                             args.irls_max_iters, args.irls_eps)
            if not ok2: break
            n, d = plane2['n'], float(plane2['d'])
            growth = 0.0 if prev_sz==0 else (union.size - prev_sz)/max(1, prev_sz)
            curr = union
            if growth < args.em_min_growth: break
            prev_sz = union.size

        if curr.size >= args.min_faces_patch and float(FA[curr].sum()) >= args.min_area_patch:
            taken_label[curr] = True
            label_patches.append(dict(
                faces=curr, n=n, d=float(d), area=float(FA[curr].sum()),
                label_int=int(lbl), label_raw=label_raw, alg="RG-Strict-A"
            ))

    qualA = dict(p95_final_max=args.p95_final_max, inlier_frac_min=args.inlier_frac_min,
                 normal_p95_deg_max=args.normal_p95_deg_max,
                 thickness_max=args.thickness_max_mul*args.sweep_dist_m,
                 min_width_m=args.min_width_m, fill_frac_min=args.fill_frac_min)
    
    label_patches = merge_within_label(label_patches, F, V, FA, Cface,
                                       indptr, indices, nfaces,
                                       merge_theta_deg=args.merge_theta_deg,
                                       merge_dist_m=args.merge_dist_m,
                                       sweep_normal_deg=args.sweep_normal_deg,
                                       sweep_dist_m=args.sweep_dist_m,
                                       sweep_frac_vertices=args.sweep_frac_vertices,
                                       faces_in_label=faces_in_label,
                                       FN=FN, face_adj_lists=face_adj_lists,
                                       qual_params=qualA, gate_mode=args.gate_mode,
                                       irls_max_iters=args.irls_max_iters,
                                       irls_eps=args.irls_eps)

    qualSat = dict(p95_final_max=args.p95_final_max, inlier_frac_min=args.inlier_frac_min,
                   normal_p95_deg_max=args.sat_normal_p95_deg_max,
                   thickness_max=args.sat_thickness_max_mul*args.sat_dist_m,
                   min_width_m=args.sat_min_width_m, fill_frac_min=args.sat_fill_frac_min)
    
    label_patches = saturate_within_label(label_patches, faces_in_label, F, V, FN, FA, face_adj_lists, Cface,
                                          indptr, indices, nfaces,
                                          sat_normal_deg=args.sat_normal_deg, sat_dist_m=args.sat_dist_m,
                                          sat_frac_vertices=args.sat_frac_vertices,
                                          qual_params=qualSat, rounds=args.sat_rounds,
                                          gate_mode=args.gate_mode,
                                          irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps)
    return label_patches

def process_label_with_tiers(lbl: int,
                             F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                             FN: np.ndarray, FN_gate: np.ndarray,
                             face_adj_lists: List[np.ndarray],
                             labels_f: np.ndarray, int2raw: Dict[int,str],
                             Cface: np.ndarray, indptr: np.ndarray, indices: np.ndarray, nfaces: int,
                             args,
                             faces_override: Optional[np.ndarray] = None) -> List[Dict]:
    all_faces = faces_override if faces_override is not None else np.where(labels_f==int(lbl))[0]
    all_faces = _ensure_face_index_vector(all_faces)
    if all_faces.size == 0: return []

    small_faces, normal_faces, big_faces, huge_faces = compute_face_tiers4(
        all_faces, FA,
        small_thr=args.tier_small_area_m2,
        normal_hi=args.tier_normal_hi_m2,
        big_hi=args.tier_big_hi_m2
    )

    faces_for_growth = _exclude_small_faces(all_faces, FA, args.tier_small_area_m2)

    huge_patches = seed_huge_faces_as_planes(lbl, huge_faces, F, V, FN, FA, args, int2raw)
    big_patches  = extract_big_patches_in_label(lbl, big_faces, F, V, FN, FA, face_adj_lists, Cface, args, int2raw)
    normal_patches = process_one_label_normal_only(
        lbl, F, V, FA, FN, FN_gate, face_adj_lists, labels_f, int2raw,
        Cface, indptr, indices, nfaces, args, faces_override=normal_faces
    )

    patches = huge_patches + big_patches + normal_patches
    if patches:
        qualA = dict(p95_final_max=args.p95_final_max, inlier_frac_min=args.inlier_frac_min,
                     normal_p95_deg_max=args.normal_p95_deg_max,
                     thickness_max=args.thickness_max_mul*args.sweep_dist_m,
                     min_width_m=args.min_width_m, fill_frac_min=args.fill_frac_min)
        patches = merge_within_label(
            patches, F, V, FA, Cface,
            indptr, indices, nfaces,
            merge_theta_deg=args.merge_big_theta_deg,
            merge_dist_m=args.merge_big_dist_m,
            sweep_normal_deg=args.sweep_normal_deg, sweep_dist_m=args.sweep_dist_m,
            sweep_frac_vertices=args.sweep_frac_vertices,
            faces_in_label=faces_for_growth, FN=FN, face_adj_lists=face_adj_lists,
            qual_params=qualA, gate_mode=args.gate_mode,
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps
        )
        qualSat = dict(p95_final_max=args.p95_final_max, inlier_frac_min=args.inlier_frac_min,
                       normal_p95_deg_max=args.sat_normal_p95_deg_max,
                       thickness_max=args.sat_thickness_max_mul*args.sat_dist_m,
                       min_width_m=args.sat_min_width_m, fill_frac_min=args.sat_fill_frac_min)
        patches = saturate_within_label(
            patches, faces_for_growth, F, V, FN, FA, face_adj_lists, Cface,
            indptr, indices, nfaces,
            sat_normal_deg=args.sat_normal_deg, sat_dist_m=args.sat_dist_m,
            sat_frac_vertices=args.sat_frac_vertices,
            qual_params=qualSat, rounds=max(1, args.sat_rounds//2),
            gate_mode=args.gate_mode,
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps
        )
    return patches

# -------------------- last-stage RG + rebuild (3‑point) --------------------
def _refit_plane_from_faces(face_ids: np.ndarray, F: np.ndarray, V: np.ndarray, FA: np.ndarray):
    face_ids = _ensure_face_index_vector(face_ids)
    if face_ids.size == 0: return None
    return fit_plane_from_faces_3pt(face_ids, F, V, FA)

def rebuild_planes_from_face_pid(face_pid: np.ndarray, F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                                 labels_f: np.ndarray, int2raw: Dict[int,str],
                                 irls_max_iters: int, irls_eps: float,
                                 min_area_face_m2: float = 0.0,
                                 prev_planes_meta: Optional[Dict[int, Dict]] = None):
    present = np.unique(face_pid[face_pid >= 0])
    if present.size == 0:
        return face_pid, []
    areas_by_old = {int(pid): float(FA[face_pid == int(pid)].sum()) for pid in present}
    order = sorted(present.tolist(), key=lambda x: -areas_by_old[int(x)])

    new_face_pid = -np.ones_like(face_pid)
    planes_meta: List[Dict] = []
    thr = float(min_area_face_m2)

    for new_id, old in enumerate(order):
        fidx = np.where(face_pid == int(old))[0]
        if fidx.size == 0: continue
        fidx = _ensure_face_index_vector(fidx)
        fidx_big = fidx[FA[fidx] >= thr] if thr > 0.0 else fidx
        new_face_pid[fidx] = int(new_id)

        labs = labels_f[fidx]
        vals, cnts = np.unique(labs, return_counts=True)
        lab = int(vals[cnts.argmax()])
        lab_raw = str(int2raw.get(lab, ""))

        pl = None
        if fidx_big.size >= 1:
            pl = _refit_plane_from_faces(fidx_big, F, V, FA)

        if pl is None:
            if prev_planes_meta is not None and int(old) in prev_planes_meta:
                prev = prev_planes_meta[int(old)]
                n = np.asarray(prev.get("n", [0.0, 0.0, 1.0]), dtype=float)
                d = float(prev.get("d", 0.0))
            else:
                n = np.array([0.0,0.0,1.0]); d = 0.0
        else:
            n, d = pl[:3], float(pl[3])

        planes_meta.append(dict(
            plane_id=int(new_id),
            label_int=lab,
            label_raw=lab_raw,
            faces=int(fidx.size),
            area=float(FA[fidx].sum()),
            n=[float(n[0]), float(n[1]), float(n[2])],
            d=float(d),
            alg="RG+EM+SAT+LAST-RG"
        ))
    return new_face_pid, planes_meta

def last_stage_relaxed_rg(face_pid: np.ndarray, planes_meta: List[Dict],
                          F: np.ndarray, V: np.ndarray, FA: np.ndarray, labels_f: np.ndarray,
                          FN: np.ndarray, Cface: np.ndarray, face_adj_lists: List[np.ndarray], progress: bool,
                          dist_m: float, normal_deg: float,
                          unlabeled_ratio: float, steal_factor: float,
                          int2raw: Dict[int,str],
                          irls_max_iters: int, irls_eps: float,
                          rg_iters: int = 1,
                          min_area_face_m2: float = 0.0):
    nfaces = F.shape[0]
    unique_pids = np.unique(face_pid[face_pid >= 0])
    if unique_pids.size == 0:
        return face_pid, []

    pid2label = {int(p["plane_id"]): int(p.get("label_int", -1)) for p in planes_meta}
    base_area = {int(pid): float(FA[face_pid == int(pid)].sum()) for pid in unique_pids}
    order = sorted(unique_pids.tolist(), key=lambda x: -base_area[int(x)])

    cos_last = float(np.cos(np.deg2rad(normal_deg)))
    pbar = TQDM(order, desc="Last-stage RG", disable=not progress, dynamic_ncols=True)
    thr = float(min_area_face_m2)

    for pid in pbar:
        pid = int(pid)
        A_faces = _ensure_face_index_vector(np.where(face_pid == pid)[0])
        if A_faces.size == 0: continue
        A_label = pid2label.get(pid, -1)
        raw = "" if A_label < 0 else str(int2raw.get(int(A_label), ""))
        A_verts = np.unique(F[A_faces].reshape(-1)).size
        pbar.set_postfix(pid=pid, cls=raw[:14], F=int(A_faces.size), V=int(A_verts))

        A_base_area = base_area.get(pid, float(FA[A_faces].sum()))
        pl = _refit_plane_from_faces(A_faces, F, V, FA)
        if pl is None: continue
        n, d = pl[:3], float(pl[3])

        for _ in range(max(0, rg_iters)):
            neigh = []
            for f in A_faces: neigh.append(face_adj_lists[int(f)])
            if not neigh: break
            neighbors = _ensure_face_index_vector(np.unique(np.concatenate(neigh)))
            maskA = np.zeros(nfaces, np.bool_); maskA[A_faces] = True
            neighbors = neighbors[~maskA[neighbors]]
            if neighbors.size == 0: break

            label_ok = (labels_f[neighbors] == int(A_label))
            if not np.any(label_ok): break
            cand = neighbors[label_ok]
            normal_ok = (np.abs(FN[cand] @ n) >= cos_last)
            if not np.any(normal_ok): break
            cand = cand[normal_ok]
            td = np.abs(Cface[cand] @ n + d)
            cand = cand[td <= dist_m]
            if thr > 0.0:
                cand = cand[FA[cand] >= thr]
            if cand.size == 0: break

            added_any = False
            unlabeled = cand[face_pid[cand] < 0]
            if unlabeled.size:
                comps = _components_in_subset(unlabeled, face_adj_lists, nfaces)
                for comp in comps:
                    add_area = float(FA[comp].sum())
                    if add_area <= A_base_area * float(unlabeled_ratio):
                        face_pid[comp] = pid; added_any = True

            labeled = cand[face_pid[cand] >= 0]
            if labeled.size:
                owners = np.unique(face_pid[labeled])
                for owner in owners:
                    owner = int(owner)
                    if owner == pid: continue
                    owner_label = pid2label.get(owner, -1)
                    if owner_label != A_label: continue
                    owner_area = float(FA[face_pid == owner].sum())
                    if owner_area <= 0: continue
                    if A_base_area >= steal_factor * owner_area:
                        take = labeled[face_pid[labeled] == owner]
                        if take.size:
                            face_pid[take] = pid; added_any = True

            if not added_any: break
            A_faces = _ensure_face_index_vector(np.where(face_pid == pid)[0])

    return rebuild_planes_from_face_pid(face_pid, F, V, FA, labels_f, int2raw,
                                        irls_max_iters, irls_eps,
                                        min_area_face_m2=min_area_face_m2)

# -------------------- optional small-claim --------------------
def final_small_claim_global(face_pid: np.ndarray,
                             F: np.ndarray, V: np.ndarray, FA: np.ndarray, FN: np.ndarray, Cface: np.ndarray,
                             labels_f: np.ndarray, planes_meta: List[Dict],
                             face_adj_lists: List[np.ndarray],
                             args, int2raw: Dict[int, str]):
    small_thr = float(args.tier_small_area_m2)
    nfaces = F.shape[0]

    label2pids: Dict[int, List[int]] = {}
    planes_by_id: Dict[int, Dict] = {}
    for p in planes_meta:
        pid = int(p.get("plane_id", -1))
        if pid < 0: continue
        planes_by_id[pid] = p
        lab = int(p.get("label_int", -1))
        label2pids.setdefault(lab, []).append(pid)

    cos_n = float(np.cos(np.deg2rad(args.small_claim_normal_deg)))

    pid2faceset = None
    if int(args.small_claim_require_adjacent) == 1:
        pid2faceset = {int(pid): set(np.where(face_pid == int(pid))[0].tolist())
                       for pid in np.unique(face_pid[face_pid >= 0])}

    candidates = np.where((face_pid < 0) & (labels_f >= 0) & (FA < small_thr))[0].astype(np.int32)
    added = 0
    for f in candidates:
        lab = int(labels_f[int(f)])
        pids = label2pids.get(lab, [])
        if not pids: continue
        nf = FN[int(f)]; cf = Cface[int(f)]
        best_pid = -1; best_dist = np.inf
        for pid in pids:
            p = planes_by_id.get(int(pid))
            if p is None: continue
            n = np.asarray(p.get("n", [0,0,1.0]), dtype=float)
            d = float(p.get("d", 0.0))
            if abs(nf @ n) < cos_n: continue
            dist = abs(cf @ n + d)
            if dist > float(args.small_claim_dist_m): continue
            if pid2faceset is not None:
                neigh = face_adj_lists[int(f)]; s = pid2faceset[int(pid)]
                ok = any(int(nn) in s for nn in neigh)
                if not ok: continue
            if dist < best_dist: best_dist = dist; best_pid = int(pid)
        if best_pid >= 0:
            face_pid[int(f)] = best_pid; added += 1

    if added > 0:
        PWRITE(f"[SMALL-CLAIM] Added {added} small faces to existing planes (final stage).")
        if int(args.small_claim_refit) == 1:
            face_pid, planes_meta = rebuild_planes_from_face_pid(face_pid, F, V, FA, labels_f, int2raw,
                                                                 args.irls_max_iters, args.irls_eps,
                                                                 min_area_face_m2=args.tier_small_area_m2)
    return face_pid, planes_meta

# -------------------- POST-RANSAC over ALL unlabeled faces (per label) --------------------
def post_ransac_unlabeled_simple(face_pid: np.ndarray, planes_meta: List[Dict],
                                 F: np.ndarray, V: np.ndarray, FA: np.ndarray, FN: np.ndarray, Cface: np.ndarray,
                                 labels_f: np.ndarray, face_adj_lists: List[np.ndarray],
                                 args, int2raw: Dict[int, str]):
    nfaces_total = F.shape[0]
    start_pid = int(np.max(face_pid)) + 1
    rng = np.random.default_rng(987)

    mask_left = (face_pid < 0) & (labels_f >= 0)
    cand_all = np.where(mask_left)[0].astype(np.int32)
    if cand_all.size == 0:
        return face_pid, planes_meta

    labels_cand = np.unique(labels_f[cand_all])
    pbar = TQDM(labels_cand, desc="POST-RANSAC-ALL", disable=not bool(args.progress), dynamic_ncols=True)
    for lbl in pbar:
        lbl = int(lbl)
        pbar.set_postfix(lbl=lbl)
        faces_lbl = cand_all[labels_f[cand_all] == lbl]
        if faces_lbl.size == 0: continue
        comps = _components_in_subset(faces_lbl, face_adj_lists, nfaces_total)
        label_raw = str(int2raw.get(lbl, 'unannotated'))

        for comp in comps:
            rem = _ensure_face_index_vector(comp)
            planes_made = 0
            while rem.size >= max(3, int(args.post_min_faces_patch)):
                if planes_made >= int(args.post_ransac_max_planes_per_comp): break
                pl, inliers = ransac_plane_over_faces(
                    rem, F, V, FN, Cface,
                    dist_m=float(args.post_ransac_dist_m),
                    normal_deg=float(args.post_ransac_normal_deg),
                    max_iters=int(args.post_ransac_max_iters),
                    rng=rng
                )
                if pl is None: break
                if inliers.size < int(args.post_min_faces_patch): break
                frac = float(inliers.size) / float(rem.size)
                if frac < float(args.post_ransac_min_inlier_frac): break
                area_here = float(FA[inliers].sum())
                if area_here < float(args.post_min_area_m2): break

                n, d = pl[:3], float(pl[3])
                pid = int(start_pid); start_pid += 1
                face_pid[inliers] = pid
                planes_meta.append(dict(
                    plane_id=pid, label_int=lbl, label_raw=label_raw,
                    faces=int(inliers.size), area=area_here,
                    n=[float(n[0]), float(n[1]), float(n[2])], d=d,
                    alg="POST-RANSAC-ALL"
                ))
                in_set = set(int(x) for x in inliers.tolist())
                mask = np.array([int(f) not in in_set for f in rem], dtype=bool)
                rem = rem[mask]; planes_made += 1

    return face_pid, planes_meta

# -------------------- save helpers --------------------
def _write_ply_with_face_pid_binary(vertices, faces, face_pid, labels_f, out_path, planes_json=None):
    import struct
    V = np.asarray(vertices, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    pid = np.asarray(face_pid, dtype=np.int32)
    lbl = np.asarray(labels_f, dtype=np.int32)
    if F.shape[0] != pid.shape[0]:
        raise RuntimeError("face_pid length != #faces")

    with open(out_path, "wb") as f:
        header = []
        header.append("ply")
        header.append("format binary_little_endian 1.0")
        header.append(f"element vertex {V.shape[0]}")
        header.append("property float x")
        header.append("property float y")
        header.append("property float z")
        header.append(f"element face {F.shape[0]}")
        header.append("property list uchar int vertex_indices")
        header.append("property int plane_id")
        header.append("property int label_int")
        if planes_json is not None:
            for p in planes_json:
                pid_i = int(p["plane_id"])
                n = p.get("n", [0,0,1]); d = float(p.get("d", 0.0))
                lab = int(p.get("label_int", -1)); labraw = str(p.get("label_raw",""))
                col = p.get("color_rgb", [0,0,0]); area = float(p.get("area", 0.0))
                header.append(f'comment plane_id {pid_i} label_int {lab} label_raw "{labraw}" '
                              f'n {n[0]} {n[1]} {n[2]} d {d} area {area} color {col[0]} {col[1]} {col[2]}')
        header.append("end_header\n")
        f.write(("\n".join(header)).encode("ascii"))

        f.write(V.astype("<f4").tobytes(order="C"))
        pack_u8 = struct.Struct("<B"); pack_i3 = struct.Struct("<iii"); pack_i  = struct.Struct("<i")
        for i, row in enumerate(F):
            a, b, c = int(row[0]), int(row[1]), int(row[2])
            f.write(pack_u8.pack(3))
            f.write(pack_i3.pack(a, b, c))
            f.write(pack_i.pack(int(pid[i])))
            f.write(pack_i.pack(int(lbl[i])))

def save_planes_mesh_and_json(mesh, face_pid, labels_f, planes_meta, out_dir,
                              json_name="planes.json", ply_name="planes.ply",
                              seed=7):
    os.makedirs(out_dir, exist_ok=True)
    max_pid = int(np.max(face_pid)) if face_pid.size else -1
    if max_pid >= 0:
        rng = np.random.default_rng(seed)
        palette = (rng.random((max_pid+1, 3)) * 255).astype(np.uint8)
    else:
        palette = np.zeros((0, 3), np.uint8)

    planes_json = []
    for p in planes_meta:
        pid_i = int(p["plane_id"])
        color = [int(c) for c in (palette[pid_i] if 0 <= pid_i < palette.shape[0] else np.array([0,0,0], np.uint8))]
        q = dict(p); q["color_rgb"] = color
        planes_json.append(q)

    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(planes_json, f, indent=2)
    print(f"[OUT] {json_name}")

    ply_path = os.path.join(out_dir, ply_name)
    _write_ply_with_face_pid_binary(mesh.vertices, mesh.faces, face_pid, labels_f, ply_path, planes_json=planes_json)
    print(f"[OUT] {ply_name} (binary)")

# -------------------- large-label split helpers --------------------
def split_label_faces_into_two(faces_in_label: np.ndarray,
                               Cface: np.ndarray,
                               mode: str = "axis") -> Tuple[np.ndarray, np.ndarray]:
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    C = Cface[faces_in_label]
    if faces_in_label.size == 0:
        return faces_in_label, np.empty(0, np.int32)
    if mode == "pca":
        X = C - C.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            a = Vt[0]; proj = X @ a
        except Exception:
            var = C.var(axis=0); axis = int(np.argmax(var)); proj = C[:, axis]
    else:
        var = C.var(axis=0); axis = int(np.argmax(var)); proj = C[:, axis]
    thr = np.median(proj)
    mask = proj <= thr
    A = faces_in_label[mask]; B = faces_in_label[~mask]
    if A.size == 0 or B.size == 0:
        k = faces_in_label.size // 2
        A = faces_in_label[:k]; B = faces_in_label[k:]
    return A.astype(np.int32), B.astype(np.int32)

def recursive_partition_faces(faces_in_label: np.ndarray,
                              F: np.ndarray, Cface: np.ndarray,
                              verts_threshold: int,
                              mode: str = "axis",
                              max_parts: int = 8,
                              min_faces: int = 50000) -> List[np.ndarray]:
    faces_in_label = _ensure_face_index_vector(faces_in_label)
    parts: List[np.ndarray] = []
    queue: List[np.ndarray] = [faces_in_label.astype(np.int32)]
    while queue:
        cur = queue.pop(0)
        if len(parts) + len(queue) + 1 >= max_parts:
            parts.append(cur); parts.extend(queue); queue.clear(); break
        if cur.size < 2 * int(min_faces):
            parts.append(cur); continue
        verts_cnt = int(np.unique(F[cur].reshape(-1)).size)
        if verts_cnt < int(verts_threshold):
            parts.append(cur); continue
        A, B = split_label_faces_into_two(cur, Cface, mode=mode)
        if A.size < int(min_faces) or B.size < int(min_faces):
            parts.append(cur); continue
        queue.append(A); queue.append(B)
    return [p.astype(np.int32) for p in parts if p.size > 0]

# -------------------- HDF5 DIRECT INPUT HELPERS --------------------
def _load_h5_first_dataset(path: str) -> np.ndarray:
    import h5py
    if not os.path.exists(path):
        sys.exit(f"[ERR] Missing file: {path}")
    with h5py.File(path, "r") as f:
        for v in f.values():
            return v[()]
    sys.exit(f"[ERR] No dataset found in HDF5: {path}")

def _read_group_names_csv(path: str, needed_max_id: int) -> list:
    if not os.path.exists(path):
        sys.exit(f"[ERR] groups CSV not found: {path}")
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        have_id = ("group_id" in (reader.fieldnames or []))
        if have_id:
            tmp = {}
            for row in reader:
                try:
                    gid = int(row["group_id"])
                    tmp[gid] = str(row["group_name"])
                except Exception:
                    continue
            max_id = max(needed_max_id, max(tmp.keys()) if tmp else -1)
            out = [tmp.get(i, f"unknown_{i}") for i in range(max_id+1)]
            return out
        else:
            names = [str(row["group_name"]) for row in reader]
            if needed_max_id >= len(names):
                names.extend([f"unknown_{i}" for i in range(len(names), needed_max_id+1)])
            return names

def _normalize_sem_name(name: str) -> str:
    return name

def build_mesh_and_legend_from_h5(args):
    V = _load_h5_first_dataset(args.h5_verts)
    F = _load_h5_first_dataset(args.h5_faces)
    GI = _load_h5_first_dataset(args.h5_face_groups).reshape(-1)
    if F.min() == 1:
        F = F - 1
    F = F.astype(np.int64, copy=False)
    V = V.astype(np.float32, copy=False)

    group_names = _read_group_names_csv(args.groups_csv, needed_max_id=int(GI.max()))
    raw_names = [group_names[int(i)] for i in GI]
    sem_names = [_normalize_sem_name(n) for n in raw_names] if int(args.normalize_group_names) == 1 else raw_names

    uniq = sorted(set(sem_names))
    sem_to_id = {n: i for i, n in enumerate(uniq)}
    rng = np.random.default_rng(int(args.h5_palette_seed))
    sem_to_color = {n: rng.integers(0, 256, size=3, dtype=np.uint8) for n in uniq}
    face_colors = np.array([sem_to_color[n] for n in sem_names], dtype=np.uint8)

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.visual.face_colors = face_colors

    legend_csv_path = args.legend_csv
    if (not legend_csv_path) and int(args.generate_legend_if_missing) == 1:
        legend_csv_path = os.path.join(args.out, "semantic_legend.csv")
    if legend_csv_path:
        os.makedirs(os.path.dirname(legend_csv_path), exist_ok=True)
        counts = {}
        for n in sem_names: counts[n] = counts.get(n, 0) + 1
        with open(legend_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["semantic_id","semantic_name","R","G","B","face_count"])
            writer.writeheader()
            for name in uniq:
                r,g,b = map(int, sem_to_color[name])
                writer.writerow({
                    "semantic_id": sem_to_id[name],
                    "semantic_name": name,
                    "R": r, "G": g, "B": b,
                    "face_count": counts.get(name, 0)
                })
        print(f"[OUT] semantic_legend.csv → {legend_csv_path}")

    return mesh, legend_csv_path

# -------------------- PARAMETRIC MERGE BY (n,d) --------------------
class _DSU:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.r = np.zeros(n, dtype=np.int8)
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return ra
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb; return rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra; return ra
        else:
            self.p[rb] = ra; self.r[ra] += 1; return ra

def _angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    c = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def merge_planes_by_params(face_pid: np.ndarray, planes_meta: List[Dict],
                           F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                           labels_f: np.ndarray, int2raw: Dict[int,str],
                           normal_deg: float = 1.0, d_tol: float = 0.01,
                           scope: str = "label",
                           refit: bool = True,
                           irls_max_iters: int = 8, irls_eps: float = 1e-6,
                           min_area_face_m2: float = 0.0,
                           debug: bool = False):
    pids = [int(p["plane_id"]) for p in planes_meta]
    if not pids:
        return face_pid, planes_meta
    pid2idx = {pid:i for i,pid in enumerate(pids)}
    N = np.zeros((len(pids), 3), np.float64)
    D = np.zeros((len(pids),), np.float64)
    L = np.zeros((len(pids),), np.int32)
    for i, p in enumerate(planes_meta):
        n0 = np.asarray(p.get("n", [0,0,1.0]), dtype=np.float64)
        d0 = float(p.get("d", 0.0))
        n, d = _normalize_nd(n0, d0)
        N[i], D[i] = n, d
        L[i] = int(p.get("label_int", -1))

    dsu = _DSU(len(pids))
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            if scope == "label" and L[i] != L[j]:
                continue
            ang = _angle_deg(N[i], N[j])          # true angle in degrees
            if ang > float(normal_deg):
                continue
            if abs(D[i] - D[j]) > float(d_tol):
                continue
            if debug:
                PWRITE(f"[PARAM-MERGE] pid={pids[i]} ↔ pid={pids[j]} "
                       f"angle={ang:.4f}°  |ΔD|={(abs(D[i]-D[j])*1000):.2f} mm")
            dsu.union(i, j)

    roots = [dsu.find(i) for i in range(len(pids))]
    unique_roots = sorted(set(roots))
    root2new = {r:i for i,r in enumerate(unique_roots)}
    pid2group = {pids[i]: root2new[roots[i]] for i in range(len(pids))}

    merged_face_pid = -np.ones_like(face_pid)
    for old_pid in np.unique(face_pid[face_pid >= 0]):
        idx = pid2idx.get(int(old_pid))
        if idx is None:
            continue
        gid = pid2group[int(old_pid)]
        merged_face_pid[face_pid == int(old_pid)] = int(gid)

    num_before = int(len(pids))
    num_after = int(len(unique_roots))
    print(f"[MERGE-PARAMS] Planes before={num_before} after={num_after} "
          f"(scope={scope}, normal≤{normal_deg}°, |ΔD|≤{d_tol} m)")

    return rebuild_planes_from_face_pid(merged_face_pid, F, V, FA, labels_f, int2raw,
                                        irls_max_iters, irls_eps,
                                        min_area_face_m2=min_area_face_m2)

# -------------------- normals.log writer (3‑point aggregate) --------------------
def write_normals_log(planes_meta: List[Dict],
                      out_dir: str,
                      filename: str,
                      int2raw: Dict[int, str],
                      stage: str,
                      include_per_plane: bool = True,
                      include_label_fits: bool = True,
                      F: Optional[np.ndarray] = None,
                      V: Optional[np.ndarray] = None,
                      FA: Optional[np.ndarray] = None,
                      labels_f: Optional[np.ndarray] = None,
                      irls_max_iters: int = 8,
                      irls_eps: float = 1e-6,
                      append: bool = False):
    """
    Write plane parameters grouped by semantic label to a text file.
    Plane equation: n·x + d = 0 (n unit, d>=0).

    NOTE: Aggregated per-label fits also use the 3‑point rule:
          pick the largest-area face of the label and compute from its 3 vertices.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode) as f:
        f.write("############################################################\n")
        f.write(f"# STAGE: {stage}\n")
        f.write("# Plane equation:  n·x + d = 0  (n is unit, d >= 0)\n")
        f.write("############################################################\n\n")

        if include_per_plane and planes_meta:
            by_label: Dict[int, List[Dict]] = {}
            for p in planes_meta:
                lab = int(p.get("label_int", -1))
                by_label.setdefault(lab, []).append(p)

            for lab in sorted(by_label.keys()):
                raw = "" if lab < 0 else str(int2raw.get(int(lab), ""))
                planes = sorted(by_label[lab], key=lambda q: -float(q.get("area", 0.0)))
                f.write(f"Label {lab} '{raw}'  planes={len(planes)}\n")
                for p in planes:
                    pid = int(p.get("plane_id", -1))
                    n0 = np.asarray(p.get("n", [0, 0, 1.0]), dtype=float)
                    d0 = float(p.get("d", 0.0))
                    n, d = _normalize_nd(n0, d0)
                    faces = int(p.get("faces", 0))
                    area = float(p.get("area", 0.0))
                    alg = str(p.get("alg", ""))
                    f.write(
                        f"  pid={pid:5d} "
                        f"n=({n[0]:+.6f},{n[1]:+.6f},{n[2]:+.6f}) "
                        f"d={d:+.6f}  faces={faces:7d}  area={area:.6f}  alg={alg}\n"
                    )
                f.write("\n")

        if include_label_fits and (F is not None) and (V is not None) and (labels_f is not None):
            uniq = np.unique(labels_f[labels_f >= 0])
            f.write("# Aggregated per-label fits (largest-area face per label; 3‑point)\n")
            for lab in sorted(uniq.tolist()):
                idx = np.where(labels_f == int(lab))[0]
                if idx.size < 1:
                    continue
                if FA is None or FA.size != F.shape[0]:
                    fid = int(idx[0])
                else:
                    fid = int(idx[np.argmax(FA[idx])])
                pl = fit_plane_3pt_from_face(fid, F, V)
                if pl is None:
                    continue
                n, d = _normalize_nd(pl[:3], float(pl[3]))
                raw = str(int2raw.get(int(lab), ""))
                f.write(
                    f"  LABEL {lab} '{raw}':  n=({n[0]:+.6f},{n[1]:+.6f},{n[2]:+.6f})  d={d:+.6f}  faces={idx.size}\n"
                )
            f.write("\n")

    print(f"[OUT] {filename}")

# -------------------- Scaling helpers --------------------
BASE_MPAU = 0.02539999969303608  # meters per asset unit for the original tuned scene

def _read_meters_per_asset_unit_from_scene_csv(path: str) -> Optional[float]:
    """
    Expect CSV:
      parameter_name,parameter_value
      meters_per_asset_unit,0.0253999996
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("parameter_name") or "").strip()
                if name == "meters_per_asset_unit":
                    val = float(row.get("parameter_value"))
                    return val
    except Exception as e:
        PWRITE(f"[WARN] Could not parse meters_per_asset_unit from {path}: {e}")
    return None

def _apply_scaling_to_args(args, mode: str, mpau_scene: float):
    """
    Scale distance and area thresholds based on mode and scene's MPAU.
    Returns a dict with 'dist' and 'area' lists of (name, old, new, factor).
    """
    if mode not in ("original", "metric", "off"):
        mode = "original"

    if mode == "off":
        print("[SCALE] mode=off → no parameter scaling.")
        return dict(dist=[], area=[], info=dict(mode=mode, Sdist=1.0, Sarea=1.0,
                                                base=BASE_MPAU, mpau=mpau_scene))

    if mpau_scene <= 0:
        # Fallbacks
        mpau_scene = BASE_MPAU if mode == "original" else 1.0

    if mode == "original":
        # thresholds tuned in *asset units* for BASE_MPAU; keep physical length
        Sdist = BASE_MPAU / mpau_scene
    else:  # metric
        # thresholds specified in meters; convert to mesh units
        Sdist = 1.0 / mpau_scene

    Sarea = Sdist * Sdist

    dist_params = [
        "rg_dist_m","sweep_dist_m","dist_thr","merge_dist_m",
        "rg_dist_m_small","sweep_dist_m_small","sat_dist_m",
        "last_dist_m","small_claim_dist_m",
        "big_ransac_dist_m","post_ransac_dist_m","merge_big_dist_m",
        "merge_params_d_m",
        "min_width_m","min_width_m_small"
    ]
    area_params = [
        "min_area_patch","min_area_patch_small","big_min_area_patch",
        "post_min_area_m2",
        "tier_small_area_m2","tier_normal_hi_m2","tier_big_hi_m2"
    ]

    changed_dist = []
    for name in dist_params:
        if hasattr(args, name):
            old = float(getattr(args, name))
            new = old * Sdist
            setattr(args, name, new)
            changed_dist.append((name, old, new, Sdist))
    changed_area = []
    for name in area_params:
        if hasattr(args, name):
            old = float(getattr(args, name))
            new = old * Sarea
            setattr(args, name, new)
            changed_area.append((name, old, new, Sarea))

    # Pretty report
    base = BASE_MPAU
    print("========== SCALE REPORT ==========")
    print(f"[SCALE] mode={mode}")
    print(f"[SCALE] BASE_MPAU={base:.12f} m/unit   MPAU_scene={mpau_scene:.12f} m/unit")
    print(f"[SCALE] S_dist={Sdist:.9f}   S_area={Sarea:.9f}")
    if not changed_dist and not changed_area:
        print("[SCALE] No parameters were updated.")
    else:
        if changed_dist:
            print("[SCALE] Distance-like parameters:")
            for (n, o, nn, s) in changed_dist:
                print(f"  - {n:24s}: {o:.9f} -> {nn:.9f} (×{s:.9f}, Δ={nn-o:+.9f})")
        if changed_area:
            print("[SCALE] Area-like parameters:")
            for (n, o, nn, s) in changed_area:
                print(f"  - {n:24s}: {o:.9f} -> {nn:.9f} (×{s:.9f}, Δ={nn-o:+.9f})")
    print("==================================")

    return dict(dist=changed_dist, area=changed_area, info=dict(mode=mode, Sdist=Sdist, Sarea=Sarea,
                                                                base=BASE_MPAU, mpau=mpau_scene))

# -------------------- main --------------------
def main(args):
    # ap = argparse.ArgumentParser("Planes with per-face labels, POST-RANSAC, final param-merge, normals.log (3‑point fits) + unit-aware scaling")

    # # I/O (PLY+legend)
    # ap.add_argument("--mesh", type=str, default="", help="mesh_by_semantic.ply (per-face RGB encodes label).")
    # ap.add_argument("--legend_csv", type=str, default="", help="semantic_legend.csv (maps RGB -> semantic_id, name)")
    # ap.add_argument("--out", required=True, help="Output directory")
    # ap.add_argument("--out_json_name", type=str, default="planes.json")
    # ap.add_argument("--out_ply_name", type=str, default="planes.ply")
    # ap.add_argument("--palette_seed", type=int, default=7)

    # # Optional HDF5 direct input
    # ap.add_argument("--h5_verts", type=str, default="", help="Path to mesh_vertices.hdf5")
    # ap.add_argument("--h5_faces", type=str, default="", help="Path to mesh_faces_vi.hdf5")
    # ap.add_argument("--h5_face_groups", type=str, default="", help="Path to mesh_faces_gi.hdf5")
    # ap.add_argument("--groups_csv", type=str, default="", help="Path to metadata_groups.csv")
    # ap.add_argument("--normalize_group_names", type=int, default=1)
    # ap.add_argument("--h5_palette_seed", type=int, default=42)
    # ap.add_argument("--generate_legend_if_missing", type=int, default=1)

    # # Progress
    # ap.add_argument("--progress", type=int, default=1, help="Show tqdm progress bars (1=yes, 0=no)")

    # # Policies
    # ap.add_argument("--policy_single_plane_labels", type=str, default="", help="ids/names to force single plane.")
    # ap.add_argument("--policy_skip_labels", type=str, default="", help="ids/names to skip entirely.")

    # # Recursive large-label split
    # ap.add_argument("--large_split_enable", type=int, default=1)
    # ap.add_argument("--large_split_verts", type=int, default=700000)
    # ap.add_argument("--large_split_mode", type=str, choices=["axis","pca"], default="axis")
    # ap.add_argument("--large_split_recursive", type=int, default=1)
    # ap.add_argument("--large_split_max_parts", type=int, default=8)
    # ap.add_argument("--large_split_min_faces", type=int, default=50000)

    # # Parallelism
    # ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    # ap.add_argument("--backend", type=str, choices=["threads","processes"], default="threads")

    # # Growth / EM / quality / merge
    # ap.add_argument("--rg_theta_deg", type=float, default=8.0)
    # ap.add_argument("--rg_dist_m", type=float, default=0.015)
    # ap.add_argument("--rg_dihedral_deg", type=float, default=55.0)
    # ap.add_argument("--rg_refit_every", type=int, default=15)
    # ap.add_argument("--rg_gate_mode", type=str, choices=["kof3","centroid","none"], default="kof3")

    # ap.add_argument("--sweep_normal_deg", type=float, default=9.0)
    # ap.add_argument("--sweep_dist_m", type=float, default=0.012)
    # ap.add_argument("--sweep_frac_vertices", type=float, default=1.0)
    # ap.add_argument("--gate_mode", type=str, choices=["kof3","centroid","none"], default="kof3")

    # ap.add_argument("--em_max_iters", type=int, default=4)
    # ap.add_argument("--em_min_growth", type=float, default=0.005)

    # ap.add_argument("--min_faces_patch", type=int, default=60)
    # ap.add_argument("--min_area_patch", type=float, default=0.10)

    # ap.add_argument("--p95_final_max", type=float, default=0.03)
    # ap.add_argument("--inlier_frac_min", type=float, default=0.80)
    # ap.add_argument("--dist_thr", type=float, default=0.012)

    # ap.add_argument("--normal_p95_deg_max", type=float, default=8.0)
    # ap.add_argument("--thickness_max_mul", type=float, default=1.6)
    # ap.add_argument("--min_width_m", type=float, default=0.06)
    # ap.add_argument("--fill_frac_min", type=float, default=0.18)

    # ap.add_argument("--merge_theta_deg", type=float, default=10.0)
    # ap.add_argument("--merge_dist_m", type=float, default=0.02)

    # # Recovery (gentle, label-only)
    # ap.add_argument("--recover_enable", type=int, default=1)
    # ap.add_argument("--rg_theta_deg_small", type=float, default=12.0)
    # ap.add_argument("--rg_dist_m_small", type=float, default=0.02)
    # ap.add_argument("--rg_dihedral_deg_small", type=float, default=60.0)
    # ap.add_argument("--rg_refit_every_small", type=int, default=20)
    # ap.add_argument("--sweep_normal_deg_small", type=float, default=12.0)
    # ap.add_argument("--sweep_dist_m_small", type=float, default=0.015)
    # ap.add_argument("--sweep_frac_vertices_small", type=float, default=0.66)
    # ap.add_argument("--em_max_iters_small", type=int, default=4)
    # ap.add_argument("--em_min_growth_small", type=float, default=0.003)
    # ap.add_argument("--min_faces_patch_small", type=int, default=28)
    # ap.add_argument("--min_area_patch_small", type=float, default=0.04)
    # ap.add_argument("--normal_p95_deg_max_small", type=float, default=12.0)
    # ap.add_argument("--thickness_max_small_mul", type=float, default=2.2)
    # ap.add_argument("--min_width_m_small", type=float, default=0.04)
    # ap.add_argument("--fill_frac_min_small", type=float, default=0.10)

    # # Saturation (label-only)
    # ap.add_argument("--sat_rounds", type=int, default=2)
    # ap.add_argument("--sat_normal_deg", type=float, default=12.0)
    # ap.add_argument("--sat_dist_m", type=float, default=0.020)
    # ap.add_argument("--sat_frac_vertices", type=float, default=0.66)
    # ap.add_argument("--sat_normal_p95_deg_max", type=float, default=10.0)
    # ap.add_argument("--sat_thickness_max_mul", type=float, default=2.0)
    # ap.add_argument("--sat_min_width_m", type=float, default=0.05)
    # ap.add_argument("--sat_fill_frac_min", type=float, default=0.15)

    # # IRLS control (compatibility only; 3‑point fits ignore)
    # ap.add_argument("--irls_max_iters", type=int, default=8)
    # ap.add_argument("--irls_eps", type=float, default=1e-6)

    # # ---- Last-stage relaxed RG controls ----
    # ap.add_argument("--last_enable", type=int, default=1)
    # ap.add_argument("--last_dist_m", type=float, default=0.020)
    # ap.add_argument("--last_normal_deg", type=float, default=18.0)
    # ap.add_argument("--last_unlabeled_ratio", type=float, default=1.0)
    # ap.add_argument("--last_steal_factor", type=float, default=5.0)
    # ap.add_argument("--last_rg_iters", type=int, default=1)

    # # ---- Absolute area thresholds for 'small' exclusion ----
    # ap.add_argument("--tier_small_area_m2", type=float, default=0.02)
    # ap.add_argument("--tier_normal_hi_m2", type=float, default=10.0)
    # ap.add_argument("--tier_big_hi_m2", type=float, default=60.0)

    # # BIG RANSAC
    # ap.add_argument("--big_ransac_enable", type=int, default=1)
    # ap.add_argument("--big_ransac_max_iters", type=int, default=300)
    # ap.add_argument("--big_ransac_dist_m", type=float, default=0.010)
    # ap.add_argument("--big_ransac_normal_deg", type=float, default=10.0)
    # ap.add_argument("--big_ransac_min_inlier_frac", type=float, default=0.50)
    # ap.add_argument("--big_min_faces_patch", type=int, default=1)
    # ap.add_argument("--big_min_area_patch", type=float, default=0.02)
    # ap.add_argument("--big_p95_final_max", type=float, default=0.05)
    # ap.add_argument("--big_inlier_frac_min", type=float, default=0.50)
    # ap.add_argument("--big_normal_p95_deg_max", type=float, default=12.0)
    # ap.add_argument("--big_thickness_max_mul", type=float, default=3.0)
    # ap.add_argument("--big_min_width_m", type=float, default=0.03)
    # ap.add_argument("--big_fill_frac_min", type=float, default=0.02)
    # ap.add_argument("--merge_big_theta_deg", type=float, default=8.0)
    # ap.add_argument("--merge_big_dist_m", type=float, default=0.015)

    # # FINAL small-claim controls (OPTIONAL)
    # ap.add_argument("--small_claim_enable", type=int, default=0)
    # ap.add_argument("--small_claim_normal_deg", type=float, default=14.0)
    # ap.add_argument("--small_claim_dist_m", type=float, default=0.012)
    # ap.add_argument("--small_claim_require_adjacent", type=int, default=0)
    # ap.add_argument("--small_claim_refit", type=int, default=0)

    # # ---- POST-RANSAC over all unlabeled faces ----
    # ap.add_argument("--post_ransac_enable", type=int, default=1)
    # ap.add_argument("--post_ransac_dist_m", type=float, default=0.012)
    # ap.add_argument("--post_ransac_normal_deg", type=float, default=12.0)
    # ap.add_argument("--post_ransac_max_iters", type=int, default=400)
    # ap.add_argument("--post_ransac_min_inlier_frac", type=float, default=0.10)
    # ap.add_argument("--post_min_faces_patch", type=int, default=30)
    # ap.add_argument("--post_min_area_m2", type=float, default=1.0)
    # ap.add_argument("--post_ransac_max_planes_per_comp", type=int, default=8)

    # # ---- FINAL PARAMETRIC MERGE ----
    # ap.add_argument("--merge_params_enable", type=int, default=1)
    # ap.add_argument("--merge_params_scope", type=str, choices=["label","global"], default="label")
    # ap.add_argument("--merge_params_normal_deg", type=float, default=1.0)
    # ap.add_argument("--merge_params_d_m", type=float, default=0.01)
    # ap.add_argument("--merge_params_refit", type=int, default=1)
    # ap.add_argument("--merge_params_debug", type=int, default=0)

    # # ---- Normals log ----
    # ap.add_argument("--normals_log_enable", type=int, default=1)
    # ap.add_argument("--normals_log_name", type=str, default="normals.log")
    # ap.add_argument("--normals_log_include_per_plane", type=int, default=1)
    # ap.add_argument("--normals_log_include_label_fits", type=int, default=1)

    # # ---- Area histogram controls ----
    # ap.add_argument("--save_area_hist", type=int, default=1)
    # ap.add_argument("--area_hist_bins", type=int, default=120)
    # ap.add_argument("--area_hist_png_name", type=str, default="face_area_hist.png")

    # # ---- NEW: Scaling controls ----
    # ap.add_argument("--scale", type=str, choices=["original","metric","off"], default="original",
    #                 help="Unit scaling mode. 'original': parameters tuned for BASE_MPAU are rescaled to scene units. "
    #                      "'metric': parameters are in meters and converted to scene units. 'off': no scaling.")
    # ap.add_argument("--scene_csv", type=str, default="",
    #                 help="Path to metadata_scene.csv (to read meters_per_asset_unit).")
    # ap.add_argument("--meters_per_asset_unit", type=float, default=-1.0,
    #                 help="Override meters_per_asset_unit; if >0, overrides --scene_csv.")

    # args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    show_bar = bool(args.progress)

    # ---- Read scene scale (MPAU) & apply parameter scaling BEFORE any processing ----
    mpau_scene = float(args.meters_per_asset_unit) if args.meters_per_asset_unit and args.meters_per_asset_unit > 0 else None
    if mpau_scene is None:
        mpau_scene = _read_meters_per_asset_unit_from_scene_csv(args.scene_csv) or -1.0
    _ = _apply_scaling_to_args(args, args.scale, float(mpau_scene))

    # ---- Load mesh (PLY+legend OR HDF5 quartet) ----
    use_ply = bool(args.mesh)
    use_h5  = bool(args.h5_verts and args.h5_faces and args.h5_face_groups and args.groups_csv)
    if not (use_ply or use_h5):
        sys.exit("[ERR] Provide either --mesh + --legend_csv, or the HDF5 quartet "
                 "(--h5_verts, --h5_faces, --h5_face_groups, --groups_csv).")

    if use_h5:
        mesh, legend_csv_path = build_mesh_and_legend_from_h5(args)
    else:
        mesh = trimesh.load(args.mesh, process=False)
        legend_csv_path = args.legend_csv

    if isinstance(mesh, trimesh.Scene):
        sys.exit("[ERR] Please pass mesh_by_semantic.ply (not glTF/GLB Scene).")
    if not isinstance(mesh, trimesh.Trimesh):
        sys.exit("[ERR] Loaded object is not a single Trimesh.]")
    if not hasattr(mesh, "visual") or mesh.visual is None or mesh.visual.face_colors is None:
        sys.exit("[ERR] Mesh needs per-face colors for legend mapping.]")
    if not legend_csv_path:
        sys.exit("[ERR] --legend_csv is required (or let the HDF5 path auto-generate it).")

    # Triangulate if needed, preserving per-face colors
    F_raw = np.asarray(mesh.faces, dtype=object)
    FC_raw = np.asarray(mesh.visual.face_colors)
    need_triang = (F_raw.dtype == object) or (F_raw.ndim != 2) or (F_raw.shape[1] != 3)
    if need_triang:
        newF, newC = [], []
        per_face_colors = (FC_raw.ndim == 2 and FC_raw.shape[0] == len(F_raw))
        for i, f in enumerate(F_raw):
            f = np.asarray(f).ravel()
            if f.size < 3: continue
            v0 = int(f[0])
            for j in range(1, f.size - 1):
                tri = (v0, int(f[j]), int(f[j+1]))
                if len({tri[0], tri[1], tri[2]}) < 3: continue
                newF.append(tri)
                if per_face_colors: newC.append(FC_raw[i])
        if len(newF) == 0:
            sys.exit("[ERR] No valid triangles after triangulation.")
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=np.asarray(newF, dtype=np.int64),
                               process=True)
        if per_face_colors and len(newC) == len(newF):
            mesh.visual.face_colors = np.asarray(newC, dtype=FC_raw.dtype)

    # drop degenerates & unreferenced
    F_fix = np.asarray(mesh.faces, dtype=np.int64)
    valid = (F_fix[:,0] != F_fix[:,1]) & (F_fix[:,1] != F_fix[:,2]) & (F_fix[:,0] != F_fix[:,2]) & (F_fix.min(axis=1) >= 0)
    if not valid.all():
        mesh.update_faces(valid)
        mesh.remove_unreferenced_vertices()

    print(f"[CLEAN] Triangulated mesh: V={mesh.vertices.shape[0]:,}  F={mesh.faces.shape[0]:,}")

    # Geometry
    V = mesh.vertices.astype(np.float32, copy=False)
    F = mesh.faces.astype(np.int32, copy=False)
    assert F.ndim == 2 and F.shape[1] == 3, f"[ERR] Faces must be triangles (N,3). Got {F.shape}"

    FA = mesh.area_faces.astype(np.float32, copy=False)
    FN = np.array(mesh.face_normals, dtype=np.float32, copy=True)
    FN /= (np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12)

    print(f"[INFO] Loaded mesh: V={V.shape[0]:,}  F={F.shape[0]:,}")

    # centroids + adjacency
    Cface = V[F].mean(axis=1)
    adj_pairs = mesh.face_adjacency
    face_adj_lists = [[] for _ in range(F.shape[0])]
    for a,b in adj_pairs.astype(np.int32):
        face_adj_lists[int(a)].append(int(b)); face_adj_lists[int(b)].append(int(a))
    face_adj_lists = [np.array(x, np.int32) if x else np.empty(0, np.int32) for x in face_adj_lists]
    indptr, indices = build_csr_from_adj_lists(face_adj_lists)
    nfaces = F.shape[0]

    # Hypersim labels (from colors)
    labels_f, int2raw = _labels_from_face_colors(mesh, legend_csv_path)

    # smooth normals for early gates
    def smooth_face_normals(FN_: np.ndarray, FA_: np.ndarray, adj: List[np.ndarray], iters=1, lam=0.5):
        N = FN_.copy()
        for _ in range(iters):
            Nn = N.copy()
            for f in range(N.shape[0]):
                nb = adj[f]
                if nb.size==0: continue
                w = FA_[nb]
                avg = (w[:,None]*N[nb]).sum(axis=0) / (w.sum()+1e-12)
                v = (1.0-lam)*N[f] + lam*avg
                nrm = np.linalg.norm(v)
                if nrm>1e-12: Nn[f] = v/nrm
            N = Nn
        return N
    FN_gate = smooth_face_normals(FN, FA, face_adj_lists, iters=1, lam=0.5)

    # Label stats (for logs)
    def build_label_stats(labels_f: np.ndarray, F: np.ndarray, FA: np.ndarray, int2raw: Dict[int, str]):
        stats = {}
        uniq = np.unique(labels_f[labels_f >= 0])
        for lbl in uniq:
            faces = np.where(labels_f == int(lbl))[0]
            verts = np.unique(F[faces].reshape(-1))
            raw = str(int2raw.get(int(lbl), ""))
            stats[int(lbl)] = {"faces": int(faces.size),
                               "verts": int(verts.size),
                               "area": float(FA[faces].sum()),
                               "raw": raw}
        return stats
    label_stats = build_label_stats(labels_f, F, FA, int2raw)
    if show_bar:
        print(f"[MESH] Labels={len(label_stats)}")

    # Policies
    def parse_label_list(expr: str, int2raw: Dict[int, str]) -> set:
        if not expr: return set()
        tokens = []
        for part in expr.replace(";", ",").split(","):
            t = part.strip()
            if t: tokens.append(t)
        if not tokens: return set()
        lower_map = {i: (str(name).lower()) for i, name in int2raw.items()}
        out = set()
        for t in tokens:
            tl = t.lower()
            if tl.isdigit():
                out.add(int(tl)); continue
            exact = [i for i, name in lower_map.items() if name == tl]
            if exact: out.update(exact); continue
            part = [i for i, name in lower_map.items() if tl in name]
            if part: out.update(part); continue
            PWRITE(f"[WARN] Label token '{t}' not matched to any class.")
        return out

    single_plane_set = parse_label_list(args.policy_single_plane_labels, int2raw)
    skip_set = parse_label_list(args.policy_skip_labels, int2raw)
    if single_plane_set & skip_set:
        both = sorted(list(single_plane_set & skip_set))
        PWRITE(f"[WARN] Labels present in both single-plane and skip sets; will be SKIPPED: {both}")

    # -------- Policy: pre-assign single-plane labels --------
    patches_all: List[Dict] = []
    if single_plane_set:
        bar_policy = TQDM(sorted(single_plane_set), desc="Policy single-plane", disable=not show_bar, dynamic_ncols=True)
        for lbl in bar_policy:
            if int(lbl) in skip_set: 
                PWRITE(f"[POLICY] Skipping label {lbl} due to skip policy."); continue
            faces_in_label = _ensure_face_index_vector(np.where(labels_f==int(lbl))[0])
            raw = str(int2raw.get(int(lbl), ''))
            bar_policy.set_postfix(id=int(lbl), cls=raw[:14], F=int(faces_in_label.size),
                                   V=int(np.unique(F[faces_in_label].reshape(-1)).size))
            if faces_in_label.size == 0:
                PWRITE(f"[POLICY] Single-plane label {lbl} has 0 faces; skipping."); continue
            faces_for_policy = _exclude_small_faces(faces_in_label, FA, args.tier_small_area_m2)
            if faces_for_policy.size == 0:
                PWRITE(f"[POLICY] Single-plane label {lbl}: all faces are small → skip."); continue
            pl = fit_plane_from_faces_3pt(faces_for_policy, F, V, FA)
            if pl is None:
                PWRITE(f"[POLICY] Could not fit plane for label {lbl}; skipping."); continue
            n, d = pl[:3], float(pl[3]); area = float(FA[faces_for_policy].sum())
            patches_all.append(dict(
                faces=faces_for_policy.astype(np.int32),
                n=n, d=d, area=area,
                label_int=int(lbl), label_raw=raw, alg="POLICY-SINGLE"
            ))
            PWRITE(f"[POLICY] {raw} (id={lbl})  F={faces_in_label.size:,}  area={area:.3f}")

    # -------- Pass A with recursive large-label splitting + tiers --------
    labels_all, counts = np.unique(labels_f[labels_f>=0], return_counts=True)
    order_idx = np.argsort(-counts); ordered_labels = labels_all[order_idx]
    process_labels = [int(x) for x in ordered_labels if int(x) not in skip_set and int(x) not in single_plane_set]

    work_items: List[Dict] = []
    split_labels: set = set()
    for lbl in process_labels:
        faces_in_label = _ensure_face_index_vector(np.where(labels_f == int(lbl))[0])
        raw = str(int2raw.get(int(lbl), ""))
        verts_count = int(np.unique(F[faces_in_label].reshape(-1)).size)
        info = {"faces": int(faces_in_label.size),
                "verts": verts_count, "area": float(FA[faces_in_label].sum()),
                "raw": raw}
        if args.large_split_enable and verts_count >= int(args.large_split_verts):
            if args.large_split_recursive:
                parts = recursive_partition_faces(
                    faces_in_label, F, Cface,
                    verts_threshold=int(args.large_split_verts),
                    mode=args.large_split_mode,
                    max_parts=int(args.large_split_max_parts),
                    min_faces=int(args.large_split_min_faces)
                )
                split_labels.add(int(lbl))
                PWRITE(f"[SPLIT] {raw} (id={int(lbl)}) verts={verts_count:,} >= {args.large_split_verts} "
                       f"→ recursive parts={len(parts)} (min_faces={args.large_split_min_faces}, max_parts={args.large_split_max_parts})")
                parts = sorted(parts, key=lambda x: -x.size)
                for part_idx, part_faces in enumerate(parts):
                    sub_verts = int(np.unique(F[part_faces].reshape(-1)).size)
                    PWRITE(f"▶ Start  {raw} (id={int(lbl)})  F={info['faces']:,}  V={info['verts']:,}  "
                           f"[subset #{part_idx+1} F={part_faces.size:,} V={sub_verts:,}]")
                    work_items.append(dict(lbl=int(lbl), faces=part_faces, info=info, subV=sub_verts))
            else:
                A, B = split_label_faces_into_two(faces_in_label, Cface, mode=args.large_split_mode)
                split_labels.add(int(lbl))
                PWRITE(f"[SPLIT] {raw} (id={int(lbl)})  verts={verts_count:,} >= {args.large_split_verts} "
                       f"→ split into A: F={A.size:,}  B: F={B.size:,}")
                work_items.append(dict(lbl=int(lbl), faces=A, info=info,
                                       subV=int(np.unique(F[A].reshape(-1)).size)))
                work_items.append(dict(lbl=int(lbl), faces=B, info=info,
                                       subV=int(np.unique(F[B].reshape(-1)).size)))
        else:
            work_items.append(dict(lbl=int(lbl), faces=None, info=info, subV=info["verts"]))

    Executor = ThreadPoolExecutor if args.backend == "threads" else ProcessPoolExecutor
    if args.jobs > 1 and len(work_items) > 1:
        with Executor(max_workers=args.jobs) as ex:
            futures, meta = [], {}
            for wi in work_items:
                lbl = int(wi["lbl"]); faces_sub = wi["faces"]; info = wi["info"]
                subV = wi.get("subV", info["verts"])
                PWRITE(f"▶ Start  {info['raw']} (id={lbl})  F={info['faces']:,}  V={info['verts']:,}"
                       + ("" if faces_sub is None else f"  [subset F={faces_sub.size:,} V={subV:,}]"))
                fut = ex.submit(process_label_with_tiers, lbl,
                                F, V, FA, FN, FN_gate,
                                face_adj_lists, labels_f, int2raw,
                                Cface, indptr, indices, nfaces, args,
                                faces_override=faces_sub)
                futures.append(fut); meta[fut] = (lbl, info, faces_sub, subV)
            pbar = TQDM(total=len(futures), desc=f"Labels (pass A+tiers, {args.backend})",
                        disable=not show_bar, dynamic_ncols=True)
            for fut in as_completed(futures):
                lbl, info, faces_sub, subV = meta[fut]
                try:
                    patches_all.extend(fut.result())
                except Exception as e:
                    PWRITE(f"[ERR] Label {info['raw']} (id={lbl}) failed: {e}")
                    PWRITE(traceback.format_exc().rstrip())
                pbar.update(1)
                pbar.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            pbar.close()
    else:
        pbarA = TQDM(work_items, desc="Labels (pass A+tiers)", disable=not show_bar, dynamic_ncols=True)
        for wi in pbarA:
            lbl = int(wi["lbl"]); faces_sub = wi["faces"]; info = wi["info"]
            subV = wi.get("subV", info["verts"])
            pbarA.set_postfix(id=lbl, cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            try:
                patches_all.extend(process_label_with_tiers(lbl,
                                                           F, V, FA, FN, FN_gate,
                                                           face_adj_lists, labels_f, int2raw,
                                                           Cface, indptr, indices, nfaces, args,
                                                           faces_override=faces_sub))
            except Exception as e:
                PWRITE(f"[ERR] Label {info['raw']} (id={lbl}) failed: {e}")
                PWRITE(traceback.format_exc().rstrip())
            if faces_sub is not None:
                PWRITE(f"[DONE]  {info['raw']} (id={lbl}) subset F={faces_sub.size:,} V={subV:,}")
        try: pbarA.close()
        except: pass

    # -------- Optional recovery on leftovers (gentle, label-only) --------
    if args.recover_enable:
        covered = np.zeros(F.shape[0], bool)
        for p in patches_all: covered[p['faces']] = True
        labels_for_recovery = process_labels
        pbar_rec = TQDM(labels_for_recovery, desc="Labels (recovery)", disable=not show_bar, dynamic_ncols=True)
        for lbl in pbar_rec:
            info = label_stats.get(int(lbl), {"faces": 0, "verts": 0, "area": 0.0, "raw": ""})
            pbar_rec.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info['faces'], V=info['verts'])

            faces_in_label = _ensure_face_index_vector(np.where(labels_f==int(lbl))[0])
            label_raw = str(int2raw.get(int(lbl),'unannotated'))
            leftover = faces_in_label[~covered[faces_in_label]]
            if leftover.size < max(12, args.min_faces_patch_small//2): continue

            leftover = _exclude_small_faces(leftover, FA, args.tier_small_area_m2)
            faces_for_growth = _exclude_small_faces(faces_in_label, FA, args.tier_small_area_m2)

            regs = grow_regions_in_label(
                leftover, F, V, FA,
                face_adj_lists=face_adj_lists, FN_gate=FN_gate, Cface=Cface,
                params=dict(rg_dihedral_deg=args.rg_dihedral_deg_small,
                            rg_theta_deg=args.rg_theta_deg_small,
                            rg_dist_m=args.rg_dist_m_small,
                            rg_refit_every=args.rg_refit_every_small,
                            min_faces_patch=max(12, args.min_faces_patch_small//2)),
                visited_mask_label=np.zeros(F.shape[0], bool),
                gate_mode=args.rg_gate_mode
            )

            taken_label = np.zeros(F.shape[0], bool)
            taken_label[labels_f!=int(lbl)] = True; taken_label[covered] = True
            areas = [FA[r].sum() for r in regs]
            order = np.argsort(-np.asarray(areas)) if areas else np.array([], dtype=int)
            regs_sorted = [regs[i] for i in order] if areas else []

            for reg in regs_sorted:
                reg = _ensure_face_index_vector(reg)
                if taken_label[reg].all(): continue
                ok, plane, _ = fit_and_quality(reg, F, V, FN, FA,
                                               args.p95_final_max, args.inlier_frac_min, args.dist_thr,
                                               args.normal_p95_deg_max_small,
                                               args.thickness_max_small_mul*args.sweep_dist_m_small,
                                               args.min_width_m_small, args.fill_frac_min_small,
                                               args.irls_max_iters, args.irls_eps)
                if not ok: continue
                n, d = plane['n'], float(plane['d'])

                prev_sz = 0; curr = reg.copy()
                for _ in range(max(1, args.em_max_iters_small)):
                    cand_mask = ~taken_label
                    sweep = sweep_inliers_label(n, d, faces_for_growth, F, V, FN, Cface,
                                                args.sweep_normal_deg_small, args.sweep_dist_m_small,
                                                args.sweep_frac_vertices_small, cand_mask,
                                                gate_mode=args.gate_mode)
                    sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=1, nfaces=nfaces)
                    union = _ensure_face_index_vector(np.concatenate([curr, sweep]))
                    ok2, plane2, _ = fit_and_quality(union, F, V, FN, FA,
                                                     args.p95_final_max, args.inlier_frac_min, args.dist_thr,
                                                     args.normal_p95_deg_max_small,
                                                     args.thickness_max_small_mul*args.sweep_dist_m_small,
                                                     args.min_width_m_small, args.fill_frac_min_small,
                                                     args.irls_max_iters, args.irls_eps)
                    if not ok2: break
                    n, d = plane2['n'], float(plane2['d'])
                    growth = 0.0 if prev_sz==0 else (union.size - prev_sz)/max(1, prev_sz)
                    curr = union
                    if growth < args.em_min_growth_small: break
                    prev_sz = union.size

                if curr.size >= args.min_faces_patch_small and float(FA[curr].sum()) >= args.min_area_patch_small:
                    taken_label[curr] = True; covered[curr] = True
                    patches_all.append(dict(
                        faces=curr, n=n, d=float(d), area=float(FA[curr].sum()),
                        label_int=int(lbl), label_raw=label_raw, alg="RG-Recover-B"
                    ))
        try: pbar_rec.close()
        except: pass

    # ----- Freeze planes: build global face_pid + planes_meta (pre-last-stage) -----
    face_pid = -np.ones(nfaces, np.int32)
    planes_meta = []
    for pid, p in enumerate(sorted(patches_all, key=lambda q: -q['area'])):
        f = _ensure_face_index_vector(p['faces'])
        face_pid[f] = pid
        n = p['n']; d = float(p['d'])
        planes_meta.append(dict(
            plane_id=pid,
            label_int=int(p.get('label_int', -1)),
            label_raw=p.get('label_raw',''),
            faces=int(f.size),
            area=float(FA[f].sum()),
            n=[float(n[0]), float(n[1]), float(n[2])],
            d=d,
            alg=p.get('alg','RG')
        ))

    # ----- LAST STAGE (relaxed RG within label) -----
    if args.last_enable and (np.max(face_pid) >= 0):
        face_pid, planes_meta = last_stage_relaxed_rg(
            face_pid, planes_meta,
            F, V, FA, labels_f,
            FN, Cface, face_adj_lists, progress=show_bar,
            dist_m=args.last_dist_m, normal_deg=args.last_normal_deg,
            unlabeled_ratio=args.last_unlabeled_ratio,
            steal_factor=args.last_steal_factor,
            int2raw=int2raw,
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps,
            rg_iters=int(args.last_rg_iters),
            min_area_face_m2=args.tier_small_area_m2
        )

    # ----- FINAL cross-split merge for labels that were split -----
    def collect_patches_from_face_pid(face_pid: np.ndarray,
                                      planes_meta: List[Dict],
                                      labels_f: np.ndarray,
                                      FA: np.ndarray) -> List[Dict]:
        patches = []
        for p in planes_meta:
            pid = int(p["plane_id"])
            fidx = np.where(face_pid == pid)[0]
            if fidx.size == 0: continue
            n = np.array(p.get("n", [0.0, 0.0, 1.0]), dtype=float)
            d = float(p.get("d", 0.0))
            lab = int(p.get("label_int", -1))
            raw = str(p.get("label_raw", ""))
            patches.append(dict(
                faces=_ensure_face_index_vector(fidx),
                n=n, d=d,
                area=float(FA[fidx].sum()),
                label_int=lab, label_raw=raw, alg="POST"
            ))
        return patches

    def face_pid_from_patches(patches: List[Dict], nfaces: int) -> np.ndarray:
        out = -np.ones(nfaces, np.int32)
        if not patches: return out
        patches_sorted = sorted(patches, key=lambda p: -float(p.get("area", 0.0)))
        pid = 0
        for p in patches_sorted:
            f = _ensure_face_index_vector(p["faces"])
            mask = out[f] < 0
            if np.any(mask):
                out[f[mask]] = pid
                p["plane_id"] = pid
                pid += 1
        return out

    def final_merge_for_split_labels(face_pid: np.ndarray,
                                     planes_meta: List[Dict],
                                     split_labels: set,
                                     labels_f: np.ndarray,
                                     F: np.ndarray, V: np.ndarray, FA: np.ndarray, FN: np.ndarray,
                                     Cface: np.ndarray, face_adj_lists: List[np.ndarray],
                                     indptr: np.ndarray, indices: np.ndarray, nfaces: int,
                                     args, int2raw: Dict[int, str]):
        if not split_labels:
            return face_pid, planes_meta
        PWRITE(f"[FINAL] Cross-split merge on labels: {sorted(list(split_labels))}")
        patches_all = collect_patches_from_face_pid(face_pid, planes_meta, labels_f, FA)
        by_label: Dict[int, List[Dict]] = {}
        for p in patches_all:
            lab = int(p.get("label_int", -1))
            by_label.setdefault(lab, []).append(p)
        merged_patches: List[Dict] = []
        for lab, group in by_label.items():
            if lab not in split_labels:
                merged_patches.extend(group); continue
            faces_in_label = _ensure_face_index_vector(np.where(labels_f == int(lab))[0])
            faces_in_label = _exclude_small_faces(faces_in_label, FA, args.tier_small_area_m2)
            qualA = dict(p95_final_max=args.p95_final_max, inlier_frac_min=args.inlier_frac_min,
                         normal_p95_deg_max=args.normal_p95_deg_max,
                         thickness_max=args.thickness_max_mul*args.sweep_dist_m,
                         min_width_m=args.min_width_m, fill_frac_min=args.fill_frac_min)
            merged = merge_within_label(
                group, F, V, FA, Cface,
                indptr, indices, nfaces,
                merge_theta_deg=args.merge_theta_deg,
                merge_dist_m=args.merge_dist_m,
                sweep_normal_deg=args.sweep_normal_deg,
                sweep_dist_m=args.sweep_dist_m,
                sweep_frac_vertices=args.sweep_frac_vertices,
                faces_in_label=faces_in_label, FN=FN, face_adj_lists=face_adj_lists,
                qual_params=qualA, gate_mode=args.gate_mode,
                irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps
            )
            merged_patches.extend(merged)
        new_face_pid = face_pid_from_patches(merged_patches, F.shape[0])
        return rebuild_planes_from_face_pid(new_face_pid, F, V, FA, labels_f, int2raw,
                                            args.irls_max_iters, args.irls_eps,
                                            min_area_face_m2=args.tier_small_area_m2)

    face_pid, planes_meta = final_merge_for_split_labels(
        face_pid, planes_meta, split_labels, labels_f,
        F, V, FA, FN, Cface, face_adj_lists,
        indptr, indices, nfaces, args, int2raw
    )

    # ----- OPTIONAL FINAL GLOBAL SMALL-FACE CLAIM -----
    if args.small_claim_enable:
        face_pid, planes_meta = final_small_claim_global(
            face_pid, F, V, FA, FN, Cface, labels_f,
            planes_meta, face_adj_lists, args, int2raw
        )

    # ----- POST-RANSAC over ALL unlabeled faces (per label) -----
    if int(args.post_ransac_enable) == 1:
        face_pid, planes_meta = post_ransac_unlabeled_simple(
            face_pid, planes_meta,
            F, V, FA, FN, Cface, labels_f,
            face_adj_lists, args, int2raw
        )
        prev_map = {int(p["plane_id"]): p for p in planes_meta}
        face_pid, planes_meta = rebuild_planes_from_face_pid(
            face_pid, F, V, FA, labels_f, int2raw,
            args.irls_max_iters, args.irls_eps,
            min_area_face_m2=args.tier_small_area_m2,
            prev_planes_meta=prev_map
        )

    # ===== normals.log (pre-merge-params snapshot) =====
    if int(args.normals_log_enable) == 1:
        write_normals_log(
            planes_meta=planes_meta,
            out_dir=args.out,
            filename=args.normals_log_name,
            int2raw=int2raw,
            stage="pre-merge-params",
            include_per_plane=bool(args.normals_log_include_per_plane),
            include_label_fits=bool(args.normals_log_include_label_fits),
            F=F, V=V, FA=FA, labels_f=labels_f,
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps,
            append=False
        )

    # ===== FINAL PARAMETRIC MERGE (A,B,C,D) =====
    if int(args.merge_params_enable) == 1 and (np.max(face_pid) >= 0):
        face_pid, planes_meta = merge_planes_by_params(
            face_pid, planes_meta,
            F, V, FA, labels_f, int2raw,
            normal_deg=float(args.merge_params_normal_deg),
            d_tol=float(args.merge_params_d_m),
            scope=str(args.merge_params_scope),
            refit=bool(args.merge_params_refit),
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps,
            min_area_face_m2=args.tier_small_area_m2,
            debug=bool(args.merge_params_debug)
        )

    # ===== normals.log (final snapshot) =====
    if int(args.normals_log_enable) == 1:
        write_normals_log(
            planes_meta=planes_meta,
            out_dir=args.out,
            filename=args.normals_log_name,
            int2raw=int2raw,
            stage="final",
            include_per_plane=bool(args.normals_log_include_per_plane),
            include_label_fits=bool(args.normals_log_include_label_fits),
            F=F, V=V, FA=FA, labels_f=labels_f,
            irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps,
            append=True
        )

    # ----- Save -----
    save_planes_mesh_and_json(
        mesh=trimesh.Trimesh(vertices=V, faces=F, process=False),
        face_pid=face_pid,
        labels_f=labels_f,
        planes_meta=planes_meta,
        out_dir=args.out,
        json_name=args.out_json_name,
        ply_name=args.out_ply_name,
        seed=args.palette_seed
    )

    print(f"[DONE] Wrote plane annotations to: {args.out}")
    print(f"         - {args.out_json_name}")
    print(f"         - {args.out_ply_name} (binary)")
    if len(planes_meta) == 0:
        print("[NOTE] No planes passed quality thresholds; consider relaxing post_ransac_* parameters.")


def run(args):
    main(args)  # if your `main()` accepts `args` as input
  

# if __name__ == "__main__":
#     main()



# python3 planes_from_mesh_hypersim.py \
#   --h5_verts /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/_detail/mesh/mesh_vertices.hdf5 \
#   --h5_faces /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/_detail/mesh/mesh_faces_vi.hdf5 \
#   --h5_face_groups /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/_detail/mesh/mesh_faces_gi.hdf5 \
#   --groups_csv /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/_detail/mesh/metadata_groups.csv \
#   --scene_csv /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/_detail/metadata_scene.csv \
#   --legend_csv /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/planes_out/semantic_legend.csv \
#   --out /Users/ahmetcanyavuz/scannetpp_gt/ai_001_002/planes_out \
#   --scale original \
#   --rg_theta_deg 8.0 --rg_dist_m 0.015 --rg_dihedral_deg 55.0 --rg_refit_every 15 \
#   --sweep_normal_deg 9.0 --sweep_dist_m 0.012 --sweep_frac_vertices 1.0 \
#   --em_max_iters 4 --em_min_growth 0.005 \
#   --min_faces_patch 60 --min_area_patch 0.10 \
#   --p95_final_max 0.03 --inlier_frac_min 0.80 --dist_thr 0.012 \
#   --normal_p95_deg_max 8.0 --thickness_max_mul 1.6 --min_width_m 0.06 --fill_frac_min 0.18 \
#   --merge_theta_deg 10.0 --merge_dist_m 0.02 \
#   --recover_enable 1 \
#   --rg_theta_deg_small 12.0 --rg_dist_m_small 0.02 --rg_dihedral_deg_small 60.0 --rg_refit_every_small 20 \
#   --sweep_normal_deg_small 12.0 --sweep_dist_m_small 0.015 --sweep_frac_vertices_small 0.66 \
#   --em_max_iters_small 4 --em_min_growth_small 0.003 \
#   --min_faces_patch_small 28 --min_area_patch_small 0.04 \
#   --normal_p95_deg_max_small 12.0 --thickness_max_small_mul 2.2 \
#   --min_width_m_small 0.04 --fill_frac_min_small 0.10 \
#   --sat_rounds 2 --sat_normal_deg 12.0 --sat_dist_m 0.020 \
#   --last_enable 1 --last_dist_m 0.020 --last_normal_deg 18.0 \
#   --last_unlabeled_ratio 1.0 --last_steal_factor 5.0 --last_rg_iters 1 \
#   --large_split_enable 1 --large_split_recursive 1 \
#   --large_split_verts 700000 --large_split_max_parts 8 --large_split_min_faces 50000 \
#   --large_split_mode axis \
#   --tier_small_area_m2 1 --tier_normal_hi_m2 10.0 --tier_big_hi_m2 60.0 \
#   --big_ransac_enable 1 --big_ransac_max_iters 300 \
#   --big_ransac_dist_m 0.010 --big_ransac_normal_deg 10.0 --big_ransac_min_inlier_frac 0.50 \
#   --small_claim_enable 0 --small_claim_refit 0 \
#   --post_ransac_enable 1 --post_min_area_m2 10.0 \
#   --post_ransac_dist_m 0.012 --post_ransac_normal_deg 12.0 \
#   --post_ransac_max_iters 400 --post_ransac_min_inlier_frac 0.10 \
#   --post_min_faces_patch 30 --post_ransac_max_planes_per_comp 8 \
#   --save_area_hist 0 --area_hist_png_name face_area_hist_abs.png \
#   --merge_params_enable 1 --merge_params_scope global --merge_params_normal_deg 1 --merge_params_d_m 0.01 --merge_params_refit 1 --merge_params_debug 0 \
#   --normals_log_enable 0 --normals_log_name normals.log \
#   --normals_log_include_per_plane 1 --normals_log_include_label_fits 1