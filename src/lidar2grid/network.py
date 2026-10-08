"""From classified points to a network: poles, spans, catenary fits, graph, clearances."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass
class Pole:
    id: int
    xy: np.ndarray
    base_z: float
    height: float
    n_points: int


@dataclass
class Conductor:
    """One wire between two structures, with its fitted catenary."""
    start_xy: np.ndarray
    end_xy: np.ndarray
    a: float                 # catenary parameter (m); larger = tighter
    s0: float
    c: float
    length: float
    sag: float               # max vertical distance below the chord (m)
    rmse: float
    n_points: int
    origin: np.ndarray = field(repr=False)
    direction: np.ndarray = field(repr=False)
    pole_ids: tuple = (-1, -1)

    def curve(self, step=0.5):
        s = np.arange(0, self.length + step, step)
        z = self.c + self.a * (np.cosh((s - self.s0) / self.a) - 1)
        xy = self.origin + s[:, None] * self.direction
        return np.c_[xy, z]


def detect_poles(xyz, hag, eps=0.8, min_points=15, min_height=4.0):
    if len(xyz) == 0:
        return []
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(xyz[:, :2])
    poles = []
    for k in sorted(set(lab) - {-1}):
        m = lab == k
        if m.sum() < min_points or hag[m].max() < min_height:
            continue
        # Use the vertical trunk (dense XY core) for position, robust to crossarm points
        xy = np.median(xyz[m, :2], axis=0)
        base = np.percentile(xyz[m, 2] - hag[m], 50)
        poles.append(Pole(len(poles), xy, float(base), float(hag[m].max()), int(m.sum())))
    return poles


def _fit_catenary(s, z):
    i = np.argmin(z)
    x0 = [max(200.0, (s.max() - s.min()) ** 2 / (8 * max(np.ptp(z), 0.05))), s[i], z[i]]

    def resid(p):
        a, s0, c = p
        return c + a * (np.cosh((s - s0) / a) - 1) - z

    r = least_squares(resid, x0, bounds=([5.0, s.min() - 500, z.min() - 50], [1e5, s.max() + 500, z.max() + 50]), loss="soft_l1", f_scale=0.1)
    return r.x, float(np.sqrt(np.mean(resid(r.x) ** 2)))


def _conductor(origin, d, s, z, L, pole_ids):
    (a, s0, c), rmse = _fit_catenary(s, z)
    ss = np.linspace(0, L, 200)
    zz = c + a * (np.cosh((ss - s0) / a) - 1)
    chord = zz[0] + (zz[-1] - zz[0]) * ss / L
    return Conductor(origin, origin + L * d, a, s0, c, L, float((chord - zz).max()), rmse, len(s),
                     origin, d, pole_ids)


def _split_parallel(perp, s, z, eps=0.3, min_points=10):
    """Separate parallel conductors in one span by lateral offset and detrended height."""
    coef = np.polyfit(s, z, 2) if len(s) > 3 else [0, 0, z.mean()]
    resid = z - np.polyval(coef, s)
    lab = DBSCAN(eps=eps, min_samples=3).fit_predict(np.c_[perp, resid])
    return [lab == k for k in sorted(set(lab) - {-1}) if (lab == k).sum() >= min_points]


def extract_conductors(wire_xyz, poles, max_span=150.0, corridor=2.5, min_points=10, coverage=0.6, min_length=5.0):
    """Assign wire points to pole-to-pole spans, split parallel conductors, fit a catenary each.

    Spans are anchored on detected structures, which is how utilities model a network.
    Wire points that fit no span are clustered and fitted as unanchored conductors;
    these usually mean a missed pole or a span leaving the tile, and are worth QA.
    """
    if len(wire_xyz) == 0:
        return []
    xy = wire_xyz[:, :2]
    owner = np.full(len(xy), -1)
    pairs = []
    if len(poles) >= 2:
        P = np.array([p.xy for p in poles])
        cand = []
        for i, j in cKDTree(P).query_pairs(max_span):
            v = P[j] - P[i]; L = np.linalg.norm(v)
            cand.append((L, i, j, v / L))
        # Shortest pairs claim points first: real spans connect neighbouring structures.
        for L, i, j, d in sorted(cand):
            n = np.array([-d[1], d[0]])
            t = (xy - P[i]) @ d / L; perp = np.abs((xy - P[i]) @ n)
            m = (owner == -1) & (t > 0.0) & (t < 1.0) & (perp < corridor)
            if m.sum() < min_points:
                continue
            # A real span is covered by wire points along most of its length
            occupied = np.unique(np.floor(t[m] * L / 2.0)).size / max(1, np.ceil(L / 2.0))
            if occupied < coverage:
                continue
            owner[m] = len(pairs)
            pairs.append((i, j, P[i], d, n, L))
    out = []
    for k, (i, j, o, d, n, L) in enumerate(pairs):
        m = owner == k
        if m.sum() < min_points:
            continue
        pts = wire_xyz[m]; s = (pts[:, :2] - o) @ d; perp = (pts[:, :2] - o) @ n
        for g in _split_parallel(perp, s, pts[:, 2], min_points=min_points):
            out.append(_conductor(o + np.median(perp[g]) * n, d, s[g], pts[g, 2], L, (i, j)))
    # Unanchored wires
    rest = wire_xyz[owner == -1]
    if len(rest) >= min_points:
        lab = DBSCAN(eps=1.5, min_samples=4).fit_predict(rest)
        for k in sorted(set(lab) - {-1}):
            pts = rest[lab == k]
            if len(pts) < min_points or np.ptp(pts[:, :2], axis=0).max() < min_length:
                continue
            ctr = pts[:, :2].mean(axis=0)
            d = np.linalg.svd(pts[:, :2] - ctr, full_matrices=False)[2][0]
            s = (pts[:, :2] - ctr) @ d
            o = ctr + s.min() * d
            out.append(_conductor(o, d, s - s.min(), pts[:, 2], float(np.ptp(s)), (-1, -1)))
    return out


def build_network(poles, conductors):
    """Graph with poles as nodes and spans (bundles of parallel conductors) as edges."""
    edges = {}
    for cd in conductors:
        i, j = cd.pole_ids
        if i >= 0 and j >= 0:
            edges.setdefault(tuple(sorted((i, j))), []).append(cd)
    return {
        "nodes": [dict(id=p.id, x=float(p.xy[0]), y=float(p.xy[1]), height=p.height) for p in poles],
        "edges": [dict(u=u, v=v, n_conductors=len(c), length=float(np.mean([x.length for x in c])),
                       max_sag=float(max(x.sag for x in c)), min_a=float(min(x.a for x in c)))
                  for (u, v), c in sorted(edges.items())],
        "unanchored_conductors": sum(1 for c in conductors if c.pole_ids == (-1, -1)),
    }


def vegetation_clearance(conductors, veg_xyz, threshold=3.0, k=5):
    """Clearance from each conductor's fitted curve to vegetation.

    Uses the distance to the k-th nearest vegetation point, so a few misclassified
    points next to the wire do not create false encroachments; real encroachment is a
    branch, i.e. many points.
    """
    if len(veg_xyz) < k:
        return [dict(min_dist=float("inf"), n_within=0) for _ in conductors]
    tree = cKDTree(veg_xyz)
    res = []
    for cd in conductors:
        curve = cd.curve()
        dist, _ = tree.query(curve, k=k)
        within = tree.query_ball_point(curve, r=threshold)
        res.append(dict(min_dist=float(dist[:, -1].min()), n_within=len(set().union(*map(set, within)))))
    return res


def to_geojson(network, conductors, clearances=None, crs=None):
    feats = []
    for n in network["nodes"]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [n["x"], n["y"]]},
                      "properties": {"kind": "pole", **n}})
    for i, cd in enumerate(conductors):
        props = {"kind": "conductor", "a": cd.a, "sag": cd.sag, "length": cd.length, "rmse": cd.rmse,
                 "poles": list(cd.pole_ids)}
        if clearances:
            props.update(clearances[i])
        feats.append({"type": "Feature", "properties": props, "geometry": {
            "type": "LineString", "coordinates": cd.curve(step=2.0).tolist()}})
    gj = {"type": "FeatureCollection", "features": feats}
    if crs:
        gj["crs"] = {"type": "name", "properties": {"name": crs}}
    return gj
