"""Ground filtering and height above ground.

A simplified Progressive Morphological Filter (Zhang et al., 2003): take the lowest
point per grid cell, apply grey-scale openings with growing windows, and keep
cells whose elevation change stays under a slope-dependent threshold. Points within
`dz` of the resulting DTM are ground.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


def _min_grid(xyz, cell):
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    ix = ((xyz[:, 0] - x0) / cell).astype(int)
    iy = ((xyz[:, 1] - y0) / cell).astype(int)
    grid = np.full((ix.max() + 1, iy.max() + 1), np.inf)
    np.minimum.at(grid, (ix, iy), xyz[:, 2])
    empty = ~np.isfinite(grid)
    if empty.any():  # fill holes with nearest valid cell
        _, (ii, jj) = ndimage.distance_transform_edt(empty, return_indices=True)
        grid = grid[ii, jj]
    return grid, (x0, y0)


def dtm_pmf(xyz, cell=1.0, windows=(3, 5, 9, 17, 33), slope=0.3, dh0=0.3, dh_max=3.0):
    """Return (dtm_grid, origin, cell)."""
    surf, origin = _min_grid(xyz, cell)
    nonground = np.zeros(surf.shape, bool)
    prev_w = 1
    for w in windows:
        opened = ndimage.grey_opening(surf, size=(w, w))
        dh_t = min(dh0 + slope * (w - prev_w) * cell, dh_max)
        nonground |= (surf - opened) > dh_t
        surf = opened
        prev_w = w
    # Rebuild the DTM from original minima at ground cells only
    raw, _ = _min_grid(xyz, cell)
    raw[nonground] = np.nan
    mask = np.isnan(raw)
    if mask.any():
        _, (ii, jj) = ndimage.distance_transform_edt(mask, return_indices=True)
        raw = raw[ii, jj]
    raw = ndimage.median_filter(raw, size=3)
    return raw, origin, cell


def height_above_ground(xyz, dtm, origin, cell):
    nx, ny = dtm.shape
    gx = origin[0] + (np.arange(nx) + 0.5) * cell
    gy = origin[1] + (np.arange(ny) + 0.5) * cell
    f = RegularGridInterpolator((gx, gy), dtm, bounds_error=False, fill_value=None)
    return xyz[:, 2] - f(xyz[:, :2])


def classify_ground(xyz, dz=0.3, **kw):
    """Return (is_ground mask, hag)."""
    dtm, origin, cell = dtm_pmf(xyz, **kw)
    hag = height_above_ground(xyz, dtm, origin, cell)
    return np.abs(hag) < dz, hag
