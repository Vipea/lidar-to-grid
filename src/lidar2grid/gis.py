"""Compare a reconstructed network with an existing GIS record (e.g. a utility's asset
register or OpenStreetMap power lines) and flag likely data errors.

GIS records are the "imperfect data": poles drift by metres, spans go missing, lines
that were rebuilt are never updated. The comparison is symmetric:
- detected spans with no GIS line nearby  -> missing from GIS
- GIS lines with no detected span nearby  -> possibly removed, undergrounded, or missed by the model
- GIS poles far from any detected pole    -> location error
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _densify(lines, step=1.0):
    pts, ids = [], []
    for k, line in enumerate(lines):
        line = np.asarray(line, float)[:, :2]
        for a, b in zip(line[:-1], line[1:]):
            n = max(2, int(np.linalg.norm(b - a) / step) + 1)
            t = np.linspace(0, 1, n)[:, None]
            pts.append(a + t * (b - a)); ids.append(np.full(n, k))
    return (np.vstack(pts), np.concatenate(ids)) if pts else (np.zeros((0, 2)), np.zeros(0, int))


def _match_fraction(src_pts, src_ids, ref_pts, tol):
    if len(ref_pts) == 0:
        return np.zeros(src_ids.max() + 1 if len(src_ids) else 0)
    d, _ = cKDTree(ref_pts).query(src_pts)
    n = src_ids.max() + 1
    return np.bincount(src_ids, weights=d < tol, minlength=n) / np.bincount(src_ids, minlength=n)


def compare(detected_spans, gis_lines, detected_poles=None, gis_poles=None, tol=3.0, min_match=0.8):
    """detected_spans / gis_lines: lists of polylines [[x, y], ...]. Returns a QA report dict."""
    dp, di = _densify(detected_spans); gp, gi = _densify(gis_lines)
    det_match = _match_fraction(dp, di, gp, tol)
    gis_match = _match_fraction(gp, gi, dp, tol)
    report = {
        "spans_detected": len(detected_spans),
        "gis_lines": len(gis_lines),
        "missing_from_gis": [int(i) for i in np.where(det_match < min_match)[0]],
        "gis_not_observed": [int(i) for i in np.where(gis_match < min_match)[0]],
        "length_precision": float(np.mean(det_match)) if len(det_match) else 0.0,
        "length_recall": float(np.mean(gis_match)) if len(gis_match) else 0.0,
    }
    if detected_poles is not None and gis_poles is not None and len(detected_poles) and len(gis_poles):
        d, idx = cKDTree(np.asarray(detected_poles)).query(np.asarray(gis_poles))
        report["gis_pole_offsets_m"] = np.round(d, 2).tolist()
        report["gis_poles_misplaced"] = [int(i) for i in np.where((d > 1.5) & (d < 15))[0]]
        report["gis_poles_unmatched"] = [int(i) for i in np.where(d >= 15)[0]]
    return report


def perturb_gis(poles_xy, rng, drift=2.0, n_misplaced=1, misplace=6.0, drop_span=True):
    """Make a realistic 'imperfect' GIS record from true pole positions, for demos and tests."""
    gp = poles_xy + rng.normal(0, drift / 3, poles_xy.shape)
    bad = rng.choice(len(gp), n_misplaced, replace=False)
    gp[bad] += rng.choice([-1, 1], (n_misplaced, 2)) * misplace
    lines = [[gp[k], gp[k + 1]] for k in range(len(gp) - 1)]
    dropped = None
    if drop_span and len(lines) > 2:
        dropped = int(rng.integers(len(lines)))
        lines.pop(dropped)
    return gp, lines, bad.tolist(), dropped
