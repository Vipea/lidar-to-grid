"""Per-point geometric features from local neighbourhoods.

Covariance eigenfeatures (Weinmann et al., 2015) at several scales capture whether a
neighbourhood looks like a line (wires, pole trunks), a plane (ground, roofs) or a
volume (vegetation). Verticality and height above ground separate poles from wires
and low from high structures.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES: list[str] = []


def _eigen_features(xyz, tree, k, chunk=200_000):
    n = len(xyz)
    out = np.empty((n, 7), np.float32)
    for start in range(0, n, chunk):
        q = xyz[start:start + chunk]
        _, idx = tree.query(q, k=k, workers=-1)
        nb = xyz[idx]                                   # (m, k, 3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / k
        w, v = np.linalg.eigh(cov)                      # ascending
        l3, l2, l1 = (np.clip(w[:, i], 1e-12, None) for i in range(3))
        e1 = v[:, :, 2]                                 # principal direction
        normal = v[:, :, 0]
        s = l1 + l2 + l3
        out[start:start + chunk] = np.c_[
            (l1 - l2) / l1,                  # linearity
            (l2 - l3) / l1,                  # planarity
            l3 / l1,                         # scattering
            np.abs(e1[:, 2]),                # principal axis verticality (1 = vertical line)
            1 - np.abs(normal[:, 2]),        # surface verticality (1 = wall)
            l3 / s,                          # change of curvature
            np.sqrt(l1),                     # extent along principal axis
        ]
    return out


def _column_features(xyz, hag, cell=1.0):
    """Vertical structure per 2D grid cell: z-range, point count, and height of the point
    relative to the cell's highest return (wires sit high in otherwise empty columns,
    pole points fill a tall, narrow column)."""
    ix = np.floor((xyz[:, 0] - xyz[:, 0].min()) / cell).astype(np.int64)
    iy = np.floor((xyz[:, 1] - xyz[:, 1].min()) / cell).astype(np.int64)
    key = ix * (iy.max() + 1) + iy
    _, inv = np.unique(key, return_inverse=True)
    m = inv.max() + 1
    hi = np.full(m, -np.inf); lo = np.full(m, np.inf); cnt = np.zeros(m)
    np.maximum.at(hi, inv, hag); np.minimum.at(lo, inv, hag); np.add.at(cnt, inv, 1)
    # points in the cell more than 1 m above ground (non-ground occupancy)
    elev = np.zeros(m); np.add.at(elev, inv, (hag > 1.0).astype(float))
    return np.c_[hi[inv] - lo[inv], hi[inv] - hag, cnt[inv], elev[inv] / cnt[inv]].astype(np.float32)


def compute_features(xyz, hag, scales=(10, 30), extra=None):
    """Return (features, names). `extra` is an optional dict of per-point arrays (e.g. intensity)."""
    tree = cKDTree(xyz)
    feats, names = [], []
    base = ["linearity", "planarity", "scattering", "axis_vert", "surf_vert", "curv", "extent"]
    for k in scales:
        feats.append(_eigen_features(xyz, tree, k)); names += [f"{b}_k{k}" for b in base]
    feats.append(_column_features(xyz, hag)); names += ["col_zrange", "col_below_top", "col_count", "col_frac_elevated"]
    feats.append(hag[:, None].astype(np.float32)); names.append("hag")
    for key, arr in (extra or {}).items():
        feats.append(np.asarray(arr, np.float32)[:, None]); names.append(key)
    return np.hstack(feats), names
