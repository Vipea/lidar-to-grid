"""Synthetic distribution-line scene for tests and demos.

Generates terrain, vegetation, buildings, wooden poles with a crossarm and three
conductors hanging as catenaries between poles. Returns xyz, labels and the
ground-truth network so every stage of the pipeline can be checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import BUILDING, GROUND, POLE, VEGETATION, WIRE


@dataclass
class Scene:
    xyz: np.ndarray            # (N, 3) float64
    labels: np.ndarray         # (N,) int
    poles: np.ndarray          # (P, 2) true pole XY
    pole_height: float
    catenary_a: float          # true catenary parameter (m)


def _terrain(x, y):
    return 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 40.0)


def catenary(s, a, s0, c):
    """Height of a catenary at along-span distance s: c + a*(cosh((s-s0)/a) - 1)."""
    return c + a * (np.cosh((s - s0) / a) - 1.0)


def make_scene(
    length: float = 300.0,
    width: float = 80.0,
    span: float = 50.0,
    pole_height: float = 10.0,
    catenary_a: float = 400.0,
    ground_density: float = 8.0,
    n_trees: int = 40,
    seed: int = 0,
) -> Scene:
    rng = np.random.default_rng(seed)
    parts, labels = [], []

    # Ground
    n = int(length * width * ground_density)
    x, y = rng.uniform(0, length, n), rng.uniform(0, width, n)
    z = _terrain(x, y) + rng.normal(0, 0.05, n)
    parts.append(np.c_[x, y, z]); labels.append(np.full(n, GROUND))

    # Line along y = width/2 with a slight kink to make it non-trivial
    pole_x = np.arange(10.0, length - 5, span)
    pole_y = width / 2 + np.where(pole_x > length / 2, 0.15 * (pole_x - length / 2), 0.0)
    poles = np.c_[pole_x, pole_y]

    arm_offsets = np.array([-0.9, 0.0, 0.9])
    for px, py in poles:
        gz = _terrain(px, py)
        # Pole: vertical cylinder of radius 0.15 m, sparse returns
        m = int(pole_height * 12)
        h = rng.uniform(0, pole_height, m)
        th = rng.uniform(0, 2 * np.pi, m)
        parts.append(np.c_[px + 0.15 * np.cos(th), py + 0.15 * np.sin(th), gz + h])
        labels.append(np.full(m, POLE))
        # Crossarm, labelled as pole (ECLAIR convention)
        t = rng.uniform(-1.1, 1.1, 30)
        parts.append(np.c_[px + rng.normal(0, 0.03, 30), py + t, gz + pole_height - 0.3 + rng.normal(0, 0.03, 30)])
        labels.append(np.full(30, POLE))

    # Conductors between consecutive poles
    for (x0, y0), (x1, y1) in zip(poles[:-1], poles[1:]):
        d = np.array([x1 - x0, y1 - y0]); L = np.linalg.norm(d); u = d / L
        nrm = np.array([-u[1], u[0]])
        z0, z1 = _terrain(x0, y0) + pole_height - 0.3, _terrain(x1, y1) + pole_height - 0.3
        for off in arm_offsets:
            m = int(L / 0.4)
            s = np.sort(rng.uniform(0.3, L - 0.3, m))
            # Catenary with endpoints at z0, z1: lowest point between, solve for s0 approx via chord tilt
            s0 = L / 2 - catenary_a * np.arcsinh((z1 - z0) / (2 * catenary_a * np.sinh(L / (2 * catenary_a))))
            c = z0 - catenary_a * (np.cosh((0 - s0) / catenary_a) - 1)
            zz = catenary(s, catenary_a, s0, c) + rng.normal(0, 0.03, m)
            xy = np.c_[x0, y0] + s[:, None] * u + off * nrm
            parts.append(np.c_[xy, zz]); labels.append(np.full(m, WIRE))

    # Trees: trunk + ellipsoidal canopy; some deliberately close to the line
    for i in range(n_trees):
        if i < 6:
            tx = rng.uniform(20, length - 20); ty = width / 2 + rng.choice([-1, 1]) * rng.uniform(3, 6)
        else:
            tx, ty = rng.uniform(0, length), rng.uniform(0, width)
            if abs(ty - width / 2) < 6:
                continue
        gz = _terrain(tx, ty); r = rng.uniform(2, 4); htop = rng.uniform(6, 14)
        m = int(np.pi * r * r * 10)
        u3 = rng.normal(size=(m, 3)); u3 /= np.linalg.norm(u3, axis=1, keepdims=True)
        rad = rng.uniform(0.3, 1.0, m)[:, None]
        pts = u3 * rad * np.array([r, r, r * 0.8])
        pts += np.array([tx, ty, gz + htop - r * 0.8])
        parts.append(pts); labels.append(np.full(m, VEGETATION))

    # Buildings: flat-roof boxes away from the line
    for _ in range(5):
        bx, by = rng.uniform(0, length - 15), rng.choice([rng.uniform(0, width / 2 - 15), rng.uniform(width / 2 + 10, width - 12)])
        w, d, hgt = rng.uniform(8, 15), rng.uniform(8, 12), rng.uniform(4, 8)
        m = int(w * d * ground_density)
        xx, yy = rng.uniform(bx, bx + w, m), rng.uniform(by, by + d, m)
        zz = _terrain(bx, by) + hgt + rng.normal(0, 0.03, m)
        parts.append(np.c_[xx, yy, zz]); labels.append(np.full(m, BUILDING))
        # remove ground under the roof (occluded)
        g = parts[0]; keep = ~((g[:, 0] > bx) & (g[:, 0] < bx + w) & (g[:, 1] > by) & (g[:, 1] < by + d))
        parts[0] = g[keep]; labels[0] = labels[0][keep]

    return Scene(np.vstack(parts), np.concatenate(labels), poles, pole_height, catenary_a)
