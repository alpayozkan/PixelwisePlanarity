#!/usr/bin/env python3
"""
Ultra-fast planes-only pipeline + Policies + LAST-STAGE RELAXED REGION GROWTH + Recursive Large-Label Split
Label-STRICT Region Growing + EM expansion + Intra-Label Saturation + Final RG
--------------------------------------------------------------------------------
- Computes planes on a semantic ScanNet-style mesh (never crosses labels).
- Outputs:
    * planes.json — per-plane parameters (n, d, area, labels, color).
    * planes.ply  — original geometry with per-face: plane_id, label_int.

Speed-ups (logic preserved for main pipeline):
- CSR adjacency + (optional) Numba local-consensus filter.
- Precomputed centroids; float32 geometry; fewer temporaries.
- Larger-label-first scheduling; optional threads/processes backend.
- Progress bars via tqdm with live label + faces/verts.

Policies:
- --policy_single_plane_labels: Fit one plane for each listed label (raw name or id).
- --policy_skip_labels: Skip processing for each listed label (no planes for them).

Recursive large-label safeguard:
- If a (sub)label has >= --large_split_verts unique vertices and --large_split_enable=1,
  we recursively split its faces into two spatial halves (axis-variance or PCA median cut)
  until all parts are below threshold or we hit --large_split_max_parts or min-size guards.
  Each part is processed independently; at the end we run a dedicated per-label merge
  across all parts to reunify compatible planes.

Final stage (relaxed normal gate + centroid distance) as region growing:
- For each plane A (within its label only), starting from A's faces, grow to adjacent faces
  satisfying |n·centroid + d| <= last_dist_m AND abs(FN·n) >= cos(last_normal_deg),
  and same semantic label. Unlabeled additions and stealing respect your given ratios.

Inputs (no rendering / no iPhone poses required):
    --mesh, --segments_json, --segments_anno, --out

Dependencies: numpy, trimesh
Optional: tqdm, numba
"""

import os, sys, json, argparse
from typing import Tuple, Dict, List, Optional
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import time
import csv
from datetime import datetime

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

# ------- numba (optional, auto-fallback) -------
NUMBA = False
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

# -------------------- segments → labels --------------------
def build_vertex_labels_from_segments(mesh: trimesh.Trimesh,
                                      seg_json_path: str,
                                      anno_json_path: str) -> Tuple[np.ndarray, Dict[int,str]]:
    if not os.path.isfile(seg_json_path): sys.exit(f"[ERR] segments.json not found: {seg_json_path}")
    if not os.path.isfile(anno_json_path): sys.exit(f"[ERR] segments_anno.json not found: {anno_json_path}")
    segs = json.load(open(seg_json_path, 'r'))
    seg_idx = np.asarray(segs['segIndices'])
    Vn = mesh.vertices.shape[0]; Fn = mesh.faces.shape[0]

    anno = json.load(open(anno_json_path, 'r'))['segGroups']
    sid2raw = {}
    for g in anno:
        raw = g.get('label','unannotated')
        for s in g.get('segments', []):
            sid2raw[int(s)] = raw

    def ids_to_raw(arr):
        out = np.empty(arr.shape[0], dtype=object)
        for i, sid in enumerate(arr): out[i] = sid2raw.get(int(sid), 'unannotated')
        return out

    # vertex-sized segIndices
    if seg_idx.shape[0] == Vn:
        raw_v = ids_to_raw(seg_idx.astype(np.int64))
        uniq = sorted(set(raw_v.tolist()))
        raw2int = {r:i for i,r in enumerate(uniq)}
        int2raw = {i:r for r,i in raw2int.items()}
        labels_v = np.array([raw2int[r] for r in raw_v], dtype=np.int32)
        return labels_v, int2raw

    # face-sized segIndices → vote per vertex
    if seg_idx.shape[0] == Fn:
        raw_f = ids_to_raw(seg_idx.astype(np.int64))
        uniq = sorted(set(raw_f.tolist()))
        raw2int = {r:i for i,r in enumerate(uniq)}
        int2raw = {i:r for r,i in raw2int.items()}
        labels_f = np.array([raw2int[r] for r in raw_f], dtype=np.int32)
        labels_v = -np.ones(Vn, np.int32)
        F = mesh.faces
        buckets = [[] for _ in range(Vn)]
        for fi,(a,b,c) in enumerate(F):
            la = labels_f[fi]
            buckets[a].append(la); buckets[b].append(la); buckets[c].append(la)
        for v in range(Vn):
            if buckets[v]:
                vals, cnts = np.unique(np.array(buckets[v], np.int32), return_counts=True)
                labels_v[v] = int(vals[cnts.argmax()])
        return labels_v, int2raw

    sys.exit("[ERR] segIndices length mismatch with mesh (#verts or #faces).")

# -------------------- plane helpers --------------------
def fit_plane_svd(P: np.ndarray):
    if P.shape[0] < 3: return None
    P64 = P.astype(np.float64, copy=False)
    c = P64.mean(axis=0)
    X = P64 - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]; n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    if d < 0: n = -n; d = -d
    return np.array([n[0], n[1], n[2], d], np.float64)

def fit_plane_irls(P: np.ndarray, max_iters=8, huber_k=1.345, eps=1e-6):
    if P.shape[0] < 3: return None
    pl = fit_plane_svd(P)
    if pl is None: return None
    n, d = pl[:3], float(pl[3])
    P64 = P.astype(np.float64, copy=False)
    for _ in range(max_iters):
        r = P64 @ n + d
        sigma = 1.4826 * (np.median(np.abs(r)) + 1e-12)
        c = huber_k * sigma + 1e-12
        w = np.ones_like(r); big = np.abs(r) > c; w[big] = c / (np.abs(r[big]) + 1e-12)
        W = w[:,None]
        mu = (W*P64).sum(axis=0) / (W.sum()+1e-12)
        X = (P64 - mu) * np.sqrt(W)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n_new = Vt[-1]; n_new /= (np.linalg.norm(n_new)+1e-12)
        d_new = -float(n_new @ mu)
        if d_new < 0: n_new = -n_new; d_new = -d_new
        if np.linalg.norm(n_new - n) < eps and abs(d_new - d) < eps:
            n, d = n_new, d_new; break
        n, d = n_new, d_new
    return np.array([n[0], n[1], n[2], d], np.float64)

def orthonormal_basis_from_normal(n: np.ndarray):
    a = np.array([1.0,0.0,0.0]) if abs(n[0]) < 0.9 else np.array([0.0,1.0,0.0])
    t1 = a - n*(a @ n); t1 /= (np.linalg.norm(t1)+1e-12)
    t2 = np.cross(n, t1); t2 /= (np.linalg.norm(t2)+1e-12)
    return t1, t2

# -------------------- label-strict sweep / gates --------------------
def sweep_inliers_label(n: np.ndarray, d: float,
                        faces_in_label: np.ndarray,
                        F: np.ndarray, V: np.ndarray, FN: np.ndarray, Cface: np.ndarray,
                        normal_deg: float, dist_m: float,
                        frac_vertices: float,
                        candidate_mask_label: np.ndarray,
                        gate_mode: str = "kof3") -> np.ndarray:
    """
    Label-restricted sweep:
      - Hemisphere + normal gate on face normals.
      - Distance gate options:
          "kof3"   : >=k of 3 triangle vertices within dist_m of plane (k=ceil(3*frac_vertices)).
          "centroid": |n·c + d| <= dist_m using face centroids.
          "none"    : no distance gate (just the normal gate).
    """
    cand = faces_in_label[candidate_mask_label[faces_in_label]]
    if cand.size == 0: return np.empty(0, np.int32)

    cos_face = float(np.cos(np.deg2rad(normal_deg)))
    dots = FN[cand] @ n
    cand = cand[(dots > 0.0) & (np.abs(dots) >= cos_face)]
    if cand.size == 0: return np.empty(0, np.int32)

    if gate_mode == "none":
        return cand.astype(np.int32)

    if gate_mode == "centroid":
        td = np.abs(Cface[cand] @ n + d)
        return cand[td <= dist_m].astype(np.int32)

    # kof3 (default)
    tri = F[cand].reshape(-1)
    dists = np.abs(V[tri] @ n + d).reshape(-1, 3)
    need = int(np.ceil(3.0*frac_vertices))
    ok = (dists <= dist_m).sum(axis=1) >= need
    return cand[ok].astype(np.int32)

# ---- CSR adjacency + (optional) numba-compiled consensus filter ----
def build_csr_from_adj_lists(adj_lists: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(adj_lists)
    lengths = np.fromiter((len(a) for a in adj_lists), count=n, dtype=np.int32)
    indptr = np.empty(n+1, np.int32)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    indices = np.empty(indptr[-1], np.int32)
    pos = 0
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
        for i in range(face_ids.shape[0]):
            mask[face_ids[i]] = 1
        keep_count = 0
        for i in range(face_ids.shape[0]):
            f = face_ids[i]
            start = indptr[f]; end = indptr[f+1]
            cnt = 0
            for k in range(start, end):
                if mask[indices[k]] != 0:
                    cnt += 1
            if cnt >= min_nbrs:
                keep_count += 1
        out = np.empty(keep_count, np.int32)
        j = 0
        for i in range(face_ids.shape[0]):
            f = face_ids[i]
            start = indptr[f]; end = indptr[f+1]
            cnt = 0
            for k in range(start, end):
                if mask[indices[k]] != 0:
                    cnt += 1
            if cnt >= min_nbrs:
                out[j] = f; j += 1
        for i in range(face_ids.shape[0]):
            mask[face_ids[i]] = 0
        return out

def filter_local_consensus(face_ids: np.ndarray,
                           indptr: np.ndarray, indices: np.ndarray,
                           min_nbrs: int, nfaces: int) -> np.ndarray:
    if face_ids.size == 0:
        return face_ids
    if NUMBA:
        return _filter_local_consensus_csr_numba(face_ids, indptr, indices, int(min_nbrs), int(nfaces))
    mask = np.zeros(nfaces, np.bool_)
    mask[face_ids] = True
    keep = []
    for f in face_ids:
        start, end = indptr[f], indptr[f+1]
        if np.count_nonzero(mask[indices[start:end]]) >= min_nbrs:
            keep.append(int(f))
    mask[face_ids] = False
    return np.array(keep, np.int32)

def split_by_plane_offset_clusters(face_ids: np.ndarray, Cface: np.ndarray, n: np.ndarray, d: float, gap=0.012):
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
    t1, t2 = orthonormal_basis_from_normal(n)
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
    if face_ids.size == 0: return False, None, {}
    verts = np.unique(F[face_ids].reshape(-1))
    pl = fit_plane_irls(V[verts], max_iters=irls_max_iters, eps=irls_eps)
    if pl is None: return False, None, {}
    n, d = pl[:3], float(pl[3])
    ok, m = eval_plane_quality(face_ids, n, d, F, V, FN, FA,
                               p95_max, inlier_frac_min, dist_thr,
                               normal_p95_deg_max, thickness_max,
                               min_width_m, fill_frac_min)
    if not ok: return False, None, m
    return True, dict(n=n, d=d), m

def grow_regions_in_label(faces_in_label: np.ndarray,
                          F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                          face_adj_lists: List[np.ndarray],
                          FN_gate: np.ndarray,
                          Cface: np.ndarray,
                          params: Dict,
                          visited_mask_label: np.ndarray,
                          gate_mode: str = "kof3") -> List[np.ndarray]:
    """Exact BFS growth inside label; same logic, faster micro-ops."""
    in_label = np.zeros(F.shape[0], bool); in_label[faces_in_label] = True
    order = faces_in_label[np.argsort(-FA[faces_in_label])]
    regs = []
    cos_dihedral = float(np.cos(np.deg2rad(params['rg_dihedral_deg'])))
    cos_theta    = float(np.cos(np.deg2rad(params['rg_theta_deg'])))

    for seed in order:
        seed = int(seed)
        if not in_label[seed] or visited_mask_label[seed]: continue
        a,b,c = F[seed]; P0 = V[[a,b,c]]
        pl = fit_plane_svd(P0)
        if pl is None: in_label[seed] = False; continue
        n, d = pl[:3], float(pl[3])
        stack = [seed]; in_label[seed] = False; reg = []
        cnt_since_refit = 0
        while stack:
            u = int(stack.pop())
            if visited_mask_label[u]: continue
            reg.append(u)
            cnt_since_refit += 1
            if cnt_since_refit >= params['rg_refit_every']:
                verts = np.unique(F[np.array(reg, np.int32)].reshape(-1))
                pl2 = fit_plane_irls(V[verts], max_iters=4, eps=1e-6)
                if pl2 is not None:
                    n, d = pl2[:3], float(pl2[3])
                cnt_since_refit = 0
            Nu = FN_gate[u]
            for v in face_adj_lists[u]:
                v = int(v)
                if not in_label[v] or visited_mask_label[v]: continue
                Nv = FN_gate[v]
                if abs(Nu @ Nv) < cos_dihedral: continue
                if abs(Nv @ n)  < cos_theta:    continue
                if gate_mode == "none":
                    pass
                elif gate_mode == "centroid":
                    if abs(Cface[v] @ n + d) > params['rg_dist_m']: continue
                else:  # kof3
                    a2,b2,c2 = F[v]
                    dist3 = np.abs(V[[a2,b2,c2]] @ n + d)
                    if (dist3 <= params['rg_dist_m']).sum() < 2: continue
                in_label[v] = False; stack.append(v)
        if len(reg) >= params['min_faces_patch']:
            regs.append(np.array(reg, np.int32))
    return regs

# -------------------- merging & saturation (use CSR consensus) --------------------
def merge_within_label(patches_label, F, V, FA, Cface,
                       indptr, indices, nfaces,
                       merge_theta_deg, merge_dist_m,
                       sweep_normal_deg, sweep_dist_m, sweep_frac_vertices,
                       faces_in_label, FN, face_adj_lists, qual_params,
                       gate_mode: str,
                       irls_max_iters: int, irls_eps: float):
    if not patches_label: return patches_label
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
                vA = np.unique(F[pi['faces']].reshape(-1))
                vB = np.unique(F[pj['faces']].reshape(-1))
                dA2B = np.percentile(np.abs(V[vA] @ n2 + d2), 85) if vA.size>0 else np.inf
                dB2A = np.percentile(np.abs(V[vB] @ n1 + d1), 85) if vB.size>0 else np.inf
                if max(dA2B, dB2A) > merge_dist_m: continue
                faces_union = np.unique(np.concatenate([pi['faces'], pj['faces']])).astype(np.int32)
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
                faces_union = np.unique(np.concatenate([faces_union, sweep])).astype(np.int32)
                ok2, plane2, _ = fit_and_quality(faces_union, F, V, FN, FA,
                                                 qual_params['p95_final_max'], qual_params['inlier_frac_min'],
                                                 sweep_dist_m, qual_params['normal_p95_deg_max'],
                                                 qual_params['thickness_max'], qual_params['min_width_m'],
                                                 qual_params['fill_frac_min'],
                                                 irls_max_iters, irls_eps)
                if ok2:
                    pi = dict(faces=faces_union, n=plane2['n'], d=float(plane2['d']),
                              area=float(FA[faces_union].sum()),
                              label_int=pi['label_int'], label_raw=pi['label_raw'], alg="RG-LabelMerge")
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
    for _ in range(max(1, rounds)):
        patches_label.sort(key=lambda p: -p['area'])
        for p in patches_label:
            n, d = p['n'], float(p['d'])
            cand_mask = np.ones(F.shape[0], bool)
            sweep = sweep_inliers_label(n, d, faces_in_label, F, V, FN, Cface,
                                        sat_normal_deg, sat_dist_m, sat_frac_vertices, cand_mask,
                                        gate_mode=gate_mode)
            sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=1, nfaces=nfaces)
            union = np.unique(np.concatenate([p['faces'], sweep])).astype(np.int32)
            ok, plane2, _ = fit_and_quality(union, F, V, FN, FA,
                                            qual_params['p95_final_max'], qual_params['inlier_frac_min'],
                                            sat_dist_m, qual_params['normal_p95_deg_max'],
                                            qual_params['thickness_max'], qual_params['min_width_m'],
                                            qual_params['fill_frac_min'],
                                            irls_max_iters, irls_eps)
            if ok:
                p['faces'] = union
                p['n'] = plane2['n']; p['d'] = float(plane2['d'])
                p['area'] = float(FA[union].sum())
        # resolve overlaps greedily
        face2pid = -np.ones(F.shape[0], np.int32)
        patches_label.sort(key=lambda p: -p['area'])
        for pid, p in enumerate(patches_label):
            f = p['faces']
            un = f[face2pid[f] < 0]
            face2pid[un] = pid
        new_list = []
        for pid, p in enumerate(patches_label):
            f = p['faces']; keep = f[face2pid[f] == pid]
            if keep.size == 0: continue
            p['faces'] = keep; p['area'] = float(FA[keep].sum())
            new_list.append(p)
        patches_label = new_list
        # light intra-label merge
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

# ---------- Save planes.json + planes.ply ----------
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
        pack_u8 = struct.Struct("<B")
        pack_i3 = struct.Struct("<iii")
        pack_i  = struct.Struct("<i")
        for i, (a, b, c) in enumerate(F):
            f.write(pack_u8.pack(3))
            f.write(pack_i3.pack(int(a), int(b), int(c)))
            f.write(pack_i.pack(int(pid[i])))
            f.write(pack_i.pack(int(lbl[i])))

def save_planes_mesh_and_json(mesh, face_pid, labels_f, planes_meta, out_dir,
                              json_name="planes.json", ply_name="planes.ply",
                              seed=1234):
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

# -------------------- utilities: label parsing + stats --------------------
def parse_label_list(expr: str, int2raw: Dict[int, str]) -> set:
    """Parse a comma/semicolon-separated list of label ids or raw names → set of int ids."""
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
        if exact:
            out.update(exact); continue
        part = [i for i, name in lower_map.items() if tl in name]
        if part:
            out.update(part); continue
        PWRITE(f"[WARN] Label token '{t}' not matched to any class.")
    return out

def build_label_stats(labels_f: np.ndarray, F: np.ndarray, FA: np.ndarray, int2raw: Dict[int, str]):
    """
    Precompute simple stats per semantic label:
      - faces (triangles), unique vertex count, total area, raw name.
    Returns: dict[label_id] -> {"faces": int, "verts": int, "area": float, "raw": str}
    """
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

# -------------------- large-label split helpers --------------------
def split_label_faces_into_two(faces_in_label: np.ndarray,
                               Cface: np.ndarray,
                               mode: str = "axis") -> Tuple[np.ndarray, np.ndarray]:
    """Split faces_in_label into two halves by spatial median (axis-variance or PCA)."""
    if faces_in_label.size == 0:
        return faces_in_label, np.empty(0, np.int32)
    C = Cface[faces_in_label]
    if mode == "pca":
        X = C - C.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            a = Vt[0]
            proj = X @ a
        except Exception:
            var = C.var(axis=0)
            axis = int(np.argmax(var))
            proj = C[:, axis]
    else:  # "axis"
        var = C.var(axis=0)
        axis = int(np.argmax(var))
        proj = C[:, axis]
    thr = np.median(proj)
    mask = proj <= thr
    A = faces_in_label[mask]
    B = faces_in_label[~mask]
    if A.size == 0 or B.size == 0:
        k = faces_in_label.size // 2
        A = faces_in_label[:k]
        B = faces_in_label[k:]
    return A.astype(np.int32), B.astype(np.int32)

def recursive_partition_faces(faces_in_label: np.ndarray,
                              F: np.ndarray, Cface: np.ndarray,
                              verts_threshold: int,
                              mode: str = "axis",
                              max_parts: int = 8,
                              min_faces: int = 50000) -> List[np.ndarray]:
    """
    Recursively split faces_in_label into up to max_parts parts.
    Stop splitting a part if:
      - unique vertex count < verts_threshold, OR
      - part has < 2*min_faces faces (can't produce two valid children), OR
      - max_parts would be exceeded.
    """
    parts: List[np.ndarray] = []
    queue: List[np.ndarray] = [faces_in_label.astype(np.int32)]
    while queue:
        cur = queue.pop(0)
        # guard if we'd exceed parts budget
        if len(parts) + len(queue) + 1 >= max_parts:
            parts.append(cur); parts.extend(queue); queue.clear(); break
        # small part guard
        if cur.size < 2 * int(min_faces):
            parts.append(cur); continue
        # vertex count on this subset
        verts_cnt = int(np.unique(F[cur].reshape(-1)).size)
        if verts_cnt < int(verts_threshold):
            parts.append(cur); continue
        # split
        A, B = split_label_faces_into_two(cur, Cface, mode=mode)
        # ensure both are usable
        if A.size < int(min_faces) or B.size < int(min_faces):
            parts.append(cur); continue
        queue.append(A); queue.append(B)
    # final sanity
    return [p.astype(np.int32) for p in parts if p.size > 0]

def collect_patches_from_face_pid(face_pid: np.ndarray,
                                  planes_meta: List[Dict],
                                  labels_f: np.ndarray,
                                  FA: np.ndarray) -> List[Dict]:
    """Turn (face_pid, planes_meta) into a list of patch dicts."""
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
            faces=fidx.astype(np.int32),
            n=n, d=d,
            area=float(FA[fidx].sum()),
            label_int=lab, label_raw=raw, alg="POST"
        ))
    return patches

def face_pid_from_patches(patches: List[Dict], nfaces: int) -> np.ndarray:
    """Assign plane ids by largest-first, avoiding overlaps."""
    out = -np.ones(nfaces, np.int32)
    if not patches: return out
    patches_sorted = sorted(patches, key=lambda p: -float(p.get("area", 0.0)))
    pid = 0
    for p in patches_sorted:
        f = p["faces"]
        mask = out[f] < 0
        if np.any(mask):
            out[f[mask]] = pid
            p["plane_id"] = pid
            pid += 1
    return out

# -------------------- per-label worker --------------------
def process_one_label(lbl: int,
                      F: np.ndarray, V: np.ndarray, FA: np.ndarray, FN: np.ndarray, FN_gate: np.ndarray,
                      face_adj_lists: List[np.ndarray], labels_f: np.ndarray, int2raw: Dict[int,str],
                      Cface: np.ndarray, indptr: np.ndarray, indices: np.ndarray, nfaces: int,
                      args,
                      faces_override: Optional[np.ndarray] = None) -> List[Dict]:
    faces_in_label = faces_override if faces_override is not None else np.where(labels_f==int(lbl))[0]
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
            union = np.unique(np.concatenate([curr, sweep])).astype(np.int32)
            clusters = split_by_plane_offset_clusters(union, Cface, n, d, gap=max(0.010, 0.5*args.sweep_dist_m))
            if clusters:
                seed_set = set(reg.tolist())
                best, best_ov = clusters[0], -1
                for cl in clusters:
                    ov = len(seed_set.intersection(set(cl.tolist())))
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

    # merge within label
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
                                       irls_max_iters=args.irls_max_iters, irls_eps=args.irls_eps)

    # saturation
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

# -------------------- LAST STAGE: relaxed region growing --------------------
def _components_in_subset(sub_idx: np.ndarray, adj_lists: List[np.ndarray], nfaces: int) -> List[np.ndarray]:
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

def _refit_plane_from_faces(face_ids: np.ndarray, F: np.ndarray, V: np.ndarray,
                            irls_max_iters: int, irls_eps: float):
    if face_ids.size == 0: return None
    verts = np.unique(F[face_ids].reshape(-1))
    return fit_plane_irls(V[verts], max_iters=irls_max_iters, eps=irls_eps)

def rebuild_planes_from_face_pid(face_pid: np.ndarray, F: np.ndarray, V: np.ndarray, FA: np.ndarray,
                                 labels_f: np.ndarray, int2raw: Dict[int,str],
                                 irls_max_iters: int, irls_eps: float):
    present = np.unique(face_pid[face_pid >= 0])
    if present.size == 0:
        return face_pid, []
    areas_by_old = {int(pid): float(FA[face_pid == int(pid)].sum()) for pid in present}
    order = sorted(present.tolist(), key=lambda x: -areas_by_old[int(x)])

    new_face_pid = -np.ones_like(face_pid)
    planes_meta: List[Dict] = []
    for new_id, old in enumerate(order):
        fidx = np.where(face_pid == int(old))[0]
        if fidx.size == 0: continue
        new_face_pid[fidx] = int(new_id)
        labs = labels_f[fidx]
        vals, cnts = np.unique(labs, return_counts=True)
        lab = int(vals[cnts.argmax()])
        lab_raw = str(int2raw.get(lab, ""))

        pl = _refit_plane_from_faces(fidx, F, V, irls_max_iters, irls_eps)
        if pl is None:
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
                          rg_iters: int = 1):
    nfaces = F.shape[0]
    unique_pids = np.unique(face_pid[face_pid >= 0])
    if unique_pids.size == 0:
        return face_pid, []

    pid2label = {int(p["plane_id"]): int(p.get("label_int", -1)) for p in planes_meta}
    base_area = {int(pid): float(FA[face_pid == int(pid)].sum()) for pid in unique_pids}
    order = sorted(unique_pids.tolist(), key=lambda x: -base_area[int(x)])

    cos_last = float(np.cos(np.deg2rad(normal_deg)))
    pbar = TQDM(order, desc="Last-stage RG", disable=not progress, dynamic_ncols=True)
    for pid in pbar:
        pid = int(pid)
        A_faces = np.where(face_pid == pid)[0]
        if A_faces.size == 0: continue
        A_label = pid2label.get(pid, -1)
        raw = "" if A_label < 0 else str(int2raw.get(int(A_label), ""))
        A_verts = np.unique(F[A_faces].reshape(-1)).size
        pbar.set_postfix(pid=pid, cls=raw[:14], F=int(A_faces.size), V=int(A_verts))

        A_base_area = base_area.get(pid, float(FA[A_faces].sum()))
        pl = _refit_plane_from_faces(A_faces, F, V, irls_max_iters, irls_eps)
        if pl is None: continue
        n, d = pl[:3], float(pl[3])

        for _ in range(max(0, rg_iters)):
            neigh = []
            for f in A_faces:
                neigh.append(face_adj_lists[int(f)])
            if not neigh: break
            neighbors = np.unique(np.concatenate(neigh)).astype(np.int32)
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
            if cand.size == 0: break

            added_any = False
            unlabeled = cand[face_pid[cand] < 0]
            if unlabeled.size:
                comps = _components_in_subset(unlabeled, face_adj_lists, nfaces)
                for comp in comps:
                    add_area = float(FA[comp].sum())
                    if add_area <= A_base_area * float(unlabeled_ratio):
                        face_pid[comp] = pid
                        added_any = True

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
                            face_pid[take] = pid
                            added_any = True

            if not added_any: break
            A_faces = np.where(face_pid == pid)[0]

    return rebuild_planes_from_face_pid(face_pid, F, V, FA, labels_f, int2raw, irls_max_iters, irls_eps)

# -------------------- final cross-split merge (for labels actually split) --------------------
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
            merged_patches.extend(group)
            continue
        faces_in_label = np.where(labels_f == int(lab))[0]
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
    return rebuild_planes_from_face_pid(new_face_pid, F, V, FA, labels_f, int2raw, args.irls_max_iters, args.irls_eps)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser("Planes-only Label-STRICT RG + EM + Saturation + Policies + Last-stage RG + Recursive Large-Split")

    # I/O
    ap.add_argument("--mesh", required=True, help="ScanNet-style semantic mesh (PLY/OBJ/GLB supported by trimesh)")
    ap.add_argument("--segments_json", required=True, help="ScanNet segments.json")
    ap.add_argument("--segments_anno", required=True, help="ScanNet segments_anno.json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--out_json_name", type=str, default="planes.json")
    ap.add_argument("--out_ply_name", type=str, default="planes.ply")
    ap.add_argument("--palette_seed", type=int, default=7)

    # Progress
    ap.add_argument("--progress", type=int, default=1, help="Show tqdm progress bars (1=yes, 0=no)")

    # Policies
    ap.add_argument("--policy_single_plane_labels", type=str, default="",
                    help="Comma/semicolon-separated list of label ids or raw names to assign as single planes.")
    ap.add_argument("--policy_skip_labels", type=str, default="",
                    help="Comma/semicolon-separated list of label ids or raw names to skip entirely.")

    # Recursive large-label split
    ap.add_argument("--large_split_enable", type=int, default=1,
                    help="If 1, labels with >= --large_split_verts unique verts are (recursively) split.")
    ap.add_argument("--large_split_verts", type=int, default=700000,
                    help="Vertex-count threshold to trigger splitting.")
    ap.add_argument("--large_split_mode", type=str, choices=["axis","pca"], default="axis",
                    help="Split mode for large labels: axis-variance or PCA median cut.")
    ap.add_argument("--large_split_recursive", type=int, default=1,
                    help="If 1, keep splitting parts until below threshold.")
    ap.add_argument("--large_split_max_parts", type=int, default=8,
                    help="Upper bound on the number of parts a label can be split into.")
    ap.add_argument("--large_split_min_faces", type=int, default=50000,
                    help="Do not split a part if it would create parts smaller than this.")

    # Parallelism
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4))),
                    help="Degree of parallelism (labels processed in parallel)")
    ap.add_argument("--backend", type=str, choices=["threads","processes"], default="threads",
                    help="Parallel backend (threads default; processes may speed Python loops)")

    # Growth / EM / quality / merge
    ap.add_argument("--rg_theta_deg", type=float, default=8.0)
    ap.add_argument("--rg_dist_m", type=float, default=0.015)
    ap.add_argument("--rg_dihedral_deg", type=float, default=55.0)
    ap.add_argument("--rg_refit_every", type=int, default=15)
    ap.add_argument("--rg_gate_mode", type=str, choices=["kof3","centroid","none"], default="kof3")

    ap.add_argument("--sweep_normal_deg", type=float, default=9.0)
    ap.add_argument("--sweep_dist_m", type=float, default=0.012)
    ap.add_argument("--sweep_frac_vertices", type=float, default=1.0)
    ap.add_argument("--gate_mode", type=str, choices=["kof3","centroid","none"], default="kof3")

    ap.add_argument("--em_max_iters", type=int, default=4)
    ap.add_argument("--em_min_growth", type=float, default=0.005)

    ap.add_argument("--min_faces_patch", type=int, default=60)
    ap.add_argument("--min_area_patch", type=float, default=0.10)

    ap.add_argument("--p95_final_max", type=float, default=0.03)
    ap.add_argument("--inlier_frac_min", type=float, default=0.80)
    ap.add_argument("--dist_thr", type=float, default=0.012)  # for quality only

    # Curvature/shape guards (strict pass)
    ap.add_argument("--normal_p95_deg_max", type=float, default=8.0)
    ap.add_argument("--thickness_max_mul", type=float, default=1.6)   # * sweep_dist_m
    ap.add_argument("--min_width_m", type=float, default=0.06)
    ap.add_argument("--fill_frac_min", type=float, default=0.18)

    # Intra-label merge
    ap.add_argument("--merge_theta_deg", type=float, default=10.0)
    ap.add_argument("--merge_dist_m", type=float, default=0.02)

    # Recovery (gentle, label-only)
    ap.add_argument("--recover_enable", type=int, default=1)
    ap.add_argument("--rg_theta_deg_small", type=float, default=12.0)
    ap.add_argument("--rg_dist_m_small", type=float, default=0.02)
    ap.add_argument("--rg_dihedral_deg_small", type=float, default=60.0)
    ap.add_argument("--rg_refit_every_small", type=int, default=20)
    ap.add_argument("--sweep_normal_deg_small", type=float, default=12.0)
    ap.add_argument("--sweep_dist_m_small", type=float, default=0.015)
    ap.add_argument("--sweep_frac_vertices_small", type=float, default=0.66)
    ap.add_argument("--em_max_iters_small", type=int, default=4)
    ap.add_argument("--em_min_growth_small", type=float, default=0.003)
    ap.add_argument("--min_faces_patch_small", type=int, default=28)
    ap.add_argument("--min_area_patch_small", type=float, default=0.04)
    ap.add_argument("--normal_p95_deg_max_small", type=float, default=12.0)
    ap.add_argument("--thickness_max_small_mul", type=float, default=2.2)  # * sweep_dist_m_small
    ap.add_argument("--min_width_m_small", type=float, default=0.04)
    ap.add_argument("--fill_frac_min_small", type=float, default=0.10)

    # Saturation (label-only)
    ap.add_argument("--sat_rounds", type=int, default=2)
    ap.add_argument("--sat_normal_deg", type=float, default=12.0)
    ap.add_argument("--sat_dist_m", type=float, default=0.020)
    ap.add_argument("--sat_frac_vertices", type=float, default=0.66)
    ap.add_argument("--sat_normal_p95_deg_max", type=float, default=10.0)
    ap.add_argument("--sat_thickness_max_mul", type=float, default=2.0)
    ap.add_argument("--sat_min_width_m", type=float, default=0.05)
    ap.add_argument("--sat_fill_frac_min", type=float, default=0.15)

    # IRLS control
    ap.add_argument("--irls_max_iters", type=int, default=8)
    ap.add_argument("--irls_eps", type=float, default=1e-6)

    # ---- Last-stage relaxed RG controls ----
    ap.add_argument("--last_enable", type=int, default=1, help="Enable final growth stage")
    ap.add_argument("--last_dist_m", type=float, default=0.020,
                    help="Centroid point-to-plane distance for last-stage inliers")
    ap.add_argument("--last_normal_deg", type=float, default=18.0,
                    help="Relaxed normal gate for last-stage (deg); larger is more permissive")
    ap.add_argument("--last_unlabeled_ratio", type=float, default=1.0,
                    help="Max ratio (added_component_area / A_base_area) for unlabeled components")
    ap.add_argument("--last_steal_factor", type=float, default=5.0,
                    help="Allow stealing from plane B iff A_base_area >= factor * area(B)")
    ap.add_argument("--last_rg_iters", type=int, default=1,
                    help="Iterations of relaxed region growing per plane (1 is recommended)")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    show_bar = bool(args.progress)

    # ---- Load mesh & geometry (float32 for speed) ----
    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh): sys.exit("[ERR] mesh is not a single Trimesh.")
    V = mesh.vertices.astype(np.float32, copy=False)
    F = mesh.faces.astype(np.int32, copy=False)
    FA = mesh.area_faces.astype(np.float32, copy=False)
    FN = np.array(mesh.face_normals, dtype=np.float32, copy=True)
    FN /= (np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12)

    print(f"[INFO] Loaded mesh: V={V.shape[0]:,}  F={F.shape[0]:,}")

    # centroids + adjacency (CSR + lists)
    Cface = V[F].mean(axis=1)  # (F,3) float32
    adj_pairs = mesh.face_adjacency
    face_adj_lists = [[] for _ in range(F.shape[0])]
    for a,b in adj_pairs.astype(np.int32):
        face_adj_lists[int(a)].append(int(b)); face_adj_lists[int(b)].append(int(a))
    face_adj_lists = [np.array(x, np.int32) if x else np.empty(0, np.int32) for x in face_adj_lists]
    indptr, indices = build_csr_from_adj_lists(face_adj_lists)
    nfaces = F.shape[0]

    # labels (per-vertex → vote per face)
    labels_v, int2raw = build_vertex_labels_from_segments(mesh, args.segments_json, args.segments_anno)
    labels_f = -np.ones(F.shape[0], np.int32)
    tri_labels = labels_v[F]
    for fi in range(F.shape[0]):
        tri = tri_labels[fi]
        tri = tri[tri>=0]
        if tri.size>0:
            vals, cnts = np.unique(tri, return_counts=True)
            labels_f[fi] = int(vals[cnts.argmax()])

    # one pass of normal smoothing for gates
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

    # --- Label stats for progress bars & policies ---
    label_stats = build_label_stats(labels_f, F, FA, int2raw)
    if show_bar:
        print(f"[MESH] Labels={len(label_stats)}")

    # Parse policies
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
                PWRITE(f"[POLICY] Skipping label {lbl} due to skip policy.")
                continue
            info = label_stats.get(int(lbl), {"faces": 0, "verts": 0, "area": 0.0, "raw": ""})
            bar_policy.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            faces_in_label = np.where(labels_f==int(lbl))[0]
            if faces_in_label.size == 0:
                PWRITE(f"[POLICY] Single-plane label {lbl} has 0 faces; skipping.")
                continue
            verts = np.unique(F[faces_in_label].reshape(-1))
            pl = fit_plane_irls(V[verts], max_iters=args.irls_max_iters, eps=args.irls_eps)
            if pl is None:
                PWRITE(f"[POLICY] Could not fit plane for label {lbl}; skipping.")
                continue
            n, d = pl[:3], float(pl[3])
            area = float(FA[faces_in_label].sum())
            raw = str(int2raw.get(int(lbl), ''))
            patches_all.append(dict(
                faces=faces_in_label.astype(np.int32),
                n=n, d=d, area=area,
                label_int=int(lbl), label_raw=raw, alg="POLICY-SINGLE"
            ))
            PWRITE(f"[POLICY] {raw} (id={lbl})  F={faces_in_label.size:,}  V={verts.size:,}  area={area:.3f}")

    # -------- Pass A (strict) per label, with recursive large-label splitting --------
    labels_all, counts = np.unique(labels_f[labels_f>=0], return_counts=True)
    order_idx = np.argsort(-counts)
    ordered_labels = labels_all[order_idx]
    process_labels = [int(x) for x in ordered_labels if int(x) not in skip_set and int(x) not in single_plane_set]

    # Build work items (label + faces subset[s])
    work_items: List[Dict] = []
    split_labels: set = set()
    for lbl in process_labels:
        faces_in_label = np.where(labels_f == int(lbl))[0]
        raw = str(int2raw.get(int(lbl), ""))
        # compute verts for the whole label
        verts_count = label_stats.get(int(lbl), {}).get("verts",
                            int(np.unique(F[faces_in_label].reshape(-1)).size))
        info = {"faces": int(faces_in_label.size),
                "verts": int(verts_count), "area": float(FA[faces_in_label].sum()),
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
                       f"→ recursive parts={len(parts)} "
                       f"(min_faces={args.large_split_min_faces}, max_parts={args.large_split_max_parts})")
                # largest-first scheduling of parts
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
                lbl = int(wi["lbl"])
                faces_sub = wi["faces"]
                info = wi["info"]
                subV = wi.get("subV", info["verts"])
                PWRITE(f"▶ Start  {info['raw']} (id={lbl})  F={info['faces']:,}  V={info['verts']:,}"
                       + ("" if faces_sub is None else f"  [subset F={faces_sub.size:,} V={subV:,}]"))
                fut = ex.submit(process_one_label, lbl,
                                F, V, FA, FN, FN_gate,
                                face_adj_lists, labels_f, int2raw,
                                Cface, indptr, indices, nfaces, args,
                                faces_override=faces_sub)
                futures.append(fut); meta[fut] = (lbl, info, faces_sub, subV)
            pbar = TQDM(total=len(futures), desc=f"Labels (pass A, {args.backend})",
                        disable=not show_bar, dynamic_ncols=True)
            for fut in as_completed(futures):
                lbl, info, faces_sub, subV = meta[fut]
                try:
                    patches_all.extend(fut.result())
                except Exception as e:
                    PWRITE(f"[ERR] Label {info['raw']} (id={lbl}) failed: {e}")
                pbar.update(1)
                pbar.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            pbar.close()
    else:
        pbarA = TQDM(work_items, desc="Labels (pass A)", disable=not show_bar, dynamic_ncols=True)
        for wi in pbarA:
            lbl = int(wi["lbl"])
            faces_sub = wi["faces"]
            info = wi["info"]
            subV = wi.get("subV", info["verts"])
            pbarA.set_postfix(id=lbl, cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            patches_all.extend(process_one_label(lbl,
                                                 F, V, FA, FN, FN_gate,
                                                 face_adj_lists, labels_f, int2raw,
                                                 Cface, indptr, indices, nfaces, args,
                                                 faces_override=faces_sub))
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
            pbar_rec.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])

            faces_in_label = np.where(labels_f==int(lbl))[0]
            label_raw = str(int2raw.get(int(lbl),'unannotated'))
            leftover = faces_in_label[~covered[faces_in_label]]
            if leftover.size < max(12, args.min_faces_patch_small//2): continue

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
                    sweep = sweep_inliers_label(n, d, faces_in_label, F, V, FN, Cface,
                                                args.sweep_normal_deg_small, args.sweep_dist_m_small,
                                                args.sweep_frac_vertices_small, cand_mask,
                                                gate_mode=args.gate_mode)
                    sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=1, nfaces=nfaces)
                    union = np.unique(np.concatenate([curr, sweep])).astype(np.int32)
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
    face_pid = -np.ones(F.shape[0], np.int32)
    planes_meta = []
    for pid, p in enumerate(sorted(patches_all, key=lambda q: -q['area'])):
        face_pid[p['faces']] = pid
        n = p['n']; d = float(p['d'])
        planes_meta.append(dict(
            plane_id=pid,
            label_int=int(p.get('label_int', -1)),
            label_raw=p.get('label_raw',''),
            faces=int(p['faces'].size),
            area=float(p['area']),
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
            rg_iters=int(args.last_rg_iters)
        )

    # ----- FINAL cross-split merge (only for labels actually split) -----
    if split_labels:
        face_pid, planes_meta = final_merge_for_split_labels(
            face_pid, planes_meta, split_labels, labels_f,
            F, V, FA, FN, Cface, face_adj_lists,
            indptr, indices, nfaces, args, int2raw
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
        print("[NOTE] No planes passed quality thresholds; consider relaxing parameters or gate_mode.")

def run(args):
    start_time = time.time()
    
    # args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    show_bar = bool(args.progress)

    # ---- Load mesh & geometry (float32 for speed) ----
    mesh = trimesh.load(args.mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh): sys.exit("[ERR] mesh is not a single Trimesh.")
    V = mesh.vertices.astype(np.float32, copy=False)
    F = mesh.faces.astype(np.int32, copy=False)
    FA = mesh.area_faces.astype(np.float32, copy=False)
    FN = np.array(mesh.face_normals, dtype=np.float32, copy=True)
    FN /= (np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12)

    print(f"[INFO] Loaded mesh: V={V.shape[0]:,}  F={F.shape[0]:,}")

    # centroids + adjacency (CSR + lists)
    Cface = V[F].mean(axis=1)  # (F,3) float32
    adj_pairs = mesh.face_adjacency
    face_adj_lists = [[] for _ in range(F.shape[0])]
    for a,b in adj_pairs.astype(np.int32):
        face_adj_lists[int(a)].append(int(b)); face_adj_lists[int(b)].append(int(a))
    face_adj_lists = [np.array(x, np.int32) if x else np.empty(0, np.int32) for x in face_adj_lists]
    indptr, indices = build_csr_from_adj_lists(face_adj_lists)
    nfaces = F.shape[0]

    # labels (per-vertex → vote per face)
    labels_v, int2raw = build_vertex_labels_from_segments(mesh, args.segments_json, args.segments_anno)
    labels_f = -np.ones(F.shape[0], np.int32)
    tri_labels = labels_v[F]
    for fi in range(F.shape[0]):
        tri = tri_labels[fi]
        tri = tri[tri>=0]
        if tri.size>0:
            vals, cnts = np.unique(tri, return_counts=True)
            labels_f[fi] = int(vals[cnts.argmax()])

    # one pass of normal smoothing for gates
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

    # --- Label stats for progress bars & policies ---
    label_stats = build_label_stats(labels_f, F, FA, int2raw)
    if show_bar:
        print(f"[MESH] Labels={len(label_stats)}")

    # Parse policies
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
                PWRITE(f"[POLICY] Skipping label {lbl} due to skip policy.")
                continue
            info = label_stats.get(int(lbl), {"faces": 0, "verts": 0, "area": 0.0, "raw": ""})
            bar_policy.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            faces_in_label = np.where(labels_f==int(lbl))[0]
            if faces_in_label.size == 0:
                PWRITE(f"[POLICY] Single-plane label {lbl} has 0 faces; skipping.")
                continue
            verts = np.unique(F[faces_in_label].reshape(-1))
            pl = fit_plane_irls(V[verts], max_iters=args.irls_max_iters, eps=args.irls_eps)
            if pl is None:
                PWRITE(f"[POLICY] Could not fit plane for label {lbl}; skipping.")
                continue
            n, d = pl[:3], float(pl[3])
            area = float(FA[faces_in_label].sum())
            raw = str(int2raw.get(int(lbl), ''))
            patches_all.append(dict(
                faces=faces_in_label.astype(np.int32),
                n=n, d=d, area=area,
                label_int=int(lbl), label_raw=raw, alg="POLICY-SINGLE"
            ))
            PWRITE(f"[POLICY] {raw} (id={lbl})  F={faces_in_label.size:,}  V={verts.size:,}  area={area:.3f}")

    # -------- Pass A (strict) per label, with recursive large-label splitting --------
    labels_all, counts = np.unique(labels_f[labels_f>=0], return_counts=True)
    order_idx = np.argsort(-counts)
    ordered_labels = labels_all[order_idx]
    process_labels = [int(x) for x in ordered_labels if int(x) not in skip_set and int(x) not in single_plane_set]

    # Build work items (label + faces subset[s])
    work_items: List[Dict] = []
    split_labels: set = set()
    for lbl in process_labels:
        faces_in_label = np.where(labels_f == int(lbl))[0]
        raw = str(int2raw.get(int(lbl), ""))
        # compute verts for the whole label
        verts_count = label_stats.get(int(lbl), {}).get("verts",
                            int(np.unique(F[faces_in_label].reshape(-1)).size))
        info = {"faces": int(faces_in_label.size),
                "verts": int(verts_count), "area": float(FA[faces_in_label].sum()),
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
                       f"→ recursive parts={len(parts)} "
                       f"(min_faces={args.large_split_min_faces}, max_parts={args.large_split_max_parts})")
                # largest-first scheduling of parts
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
                lbl = int(wi["lbl"])
                faces_sub = wi["faces"]
                info = wi["info"]
                subV = wi.get("subV", info["verts"])
                PWRITE(f"▶ Start  {info['raw']} (id={lbl})  F={info['faces']:,}  V={info['verts']:,}"
                       + ("" if faces_sub is None else f"  [subset F={faces_sub.size:,} V={subV:,}]"))
                fut = ex.submit(process_one_label, lbl,
                                F, V, FA, FN, FN_gate,
                                face_adj_lists, labels_f, int2raw,
                                Cface, indptr, indices, nfaces, args,
                                faces_override=faces_sub)
                futures.append(fut); meta[fut] = (lbl, info, faces_sub, subV)
            pbar = TQDM(total=len(futures), desc=f"Labels (pass A, {args.backend})",
                        disable=not show_bar, dynamic_ncols=True)
            for fut in as_completed(futures):
                lbl, info, faces_sub, subV = meta[fut]
                try:
                    patches_all.extend(fut.result())
                except Exception as e:
                    PWRITE(f"[ERR] Label {info['raw']} (id={lbl}) failed: {e}")
                pbar.update(1)
                pbar.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            pbar.close()
    else:
        pbarA = TQDM(work_items, desc="Labels (pass A)", disable=not show_bar, dynamic_ncols=True)
        for wi in pbarA:
            lbl = int(wi["lbl"])
            faces_sub = wi["faces"]
            info = wi["info"]
            subV = wi.get("subV", info["verts"])
            pbarA.set_postfix(id=lbl, cls=info["raw"][:14], F=info["faces"], V=info["verts"])
            patches_all.extend(process_one_label(lbl,
                                                 F, V, FA, FN, FN_gate,
                                                 face_adj_lists, labels_f, int2raw,
                                                 Cface, indptr, indices, nfaces, args,
                                                 faces_override=faces_sub))
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
            pbar_rec.set_postfix(id=int(lbl), cls=info["raw"][:14], F=info["faces"], V=info["verts"])

            faces_in_label = np.where(labels_f==int(lbl))[0]
            label_raw = str(int2raw.get(int(lbl),'unannotated'))
            leftover = faces_in_label[~covered[faces_in_label]]
            if leftover.size < max(12, args.min_faces_patch_small//2): continue

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
                    sweep = sweep_inliers_label(n, d, faces_in_label, F, V, FN, Cface,
                                                args.sweep_normal_deg_small, args.sweep_dist_m_small,
                                                args.sweep_frac_vertices_small, cand_mask,
                                                gate_mode=args.gate_mode)
                    sweep = filter_local_consensus(sweep, indptr, indices, min_nbrs=1, nfaces=nfaces)
                    union = np.unique(np.concatenate([curr, sweep])).astype(np.int32)
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
    face_pid = -np.ones(F.shape[0], np.int32)
    planes_meta = []
    for pid, p in enumerate(sorted(patches_all, key=lambda q: -q['area'])):
        face_pid[p['faces']] = pid
        n = p['n']; d = float(p['d'])
        planes_meta.append(dict(
            plane_id=pid,
            label_int=int(p.get('label_int', -1)),
            label_raw=p.get('label_raw',''),
            faces=int(p['faces'].size),
            area=float(p['area']),
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
            rg_iters=int(args.last_rg_iters)
        )

    # ----- FINAL cross-split merge (only for labels actually split) -----
    if split_labels:
        face_pid, planes_meta = final_merge_for_split_labels(
            face_pid, planes_meta, split_labels, labels_f,
            F, V, FA, FN, Cface, face_adj_lists,
            indptr, indices, nfaces, args, int2raw
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
        print("[NOTE] No planes passed quality thresholds; consider relaxing parameters or gate_mode.")

    # Save Runtime
    total_time_sec = time.time() - start_time
    scene_id = os.path.basename(os.path.normpath(args.out))  # e.g., "0a7cc12c0e"

    # Format timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare row
    runtime_row = {
        "timestamp": now_str,
        "scene_id": scene_id,
        "num_vertices": V.shape[0],
        "num_faces": F.shape[0],
        "num_planes": len(planes_meta),
        "runtime_sec": round(total_time_sec, 2)
    }

    # Path to runtimes.csv inside scene folder
    csv_path = os.path.join(args.out, "runtimes.csv")
    write_header = not os.path.exists(csv_path)

    # Write CSV (append-safe)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=runtime_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(runtime_row)

    print(f"[LOG] Saved runtime to: {csv_path}")


# if __name__ == "__main__":
#     # Optional: avoid BLAS oversubscription
#     # os.environ.setdefault("OMP_NUM_THREADS", "1")
#     # os.environ.setdefault("MKL_NUM_THREADS", "1")
#     # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#     # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#     main()


# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
