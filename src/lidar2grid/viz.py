"""Plan view and span profile plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import CLASS_NAMES

COLORS = ["#c8b89a", "#4c9a4c", "#b04a4a", "#1f5fbf", "#e08a00", "#888888"]


def plot_scene(xyz, labels, conductors, poles, gis_poles=None, path="scene.png", clearances=None):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})
    for c, name in enumerate(CLASS_NAMES):
        m = labels == c
        if m.any() and name != "ground":
            ax.scatter(xyz[m, 0], xyz[m, 1], s=0.3, c=COLORS[c], label=name, rasterized=True)
    for cd in conductors:
        cv = cd.curve(2.0)
        ax.plot(cv[:, 0], cv[:, 1], c="k", lw=0.8)
    if poles:
        P = np.array([p.xy for p in poles]); ax.scatter(P[:, 0], P[:, 1], marker="s", s=40, c="k", label="detected pole")
    if gis_poles is not None:
        ax.scatter(gis_poles[:, 0], gis_poles[:, 1], marker="x", s=60, c="r", label="GIS pole")
    ax.set_aspect("equal"); ax.set_title("Plan view")
    leg = ax.legend(loc="upper right", fontsize=8)
    for h in leg.legend_handles:
        h.set_sizes([25])

    if conductors:  # profile of the conductor closest to vegetation
        k = int(np.argmin([c["min_dist"] for c in clearances])) if clearances else 0
        cd = conductors[k]
        s_pts = (xyz[:, :2] - cd.origin) @ cd.direction
        perp = np.abs((xyz[:, :2] - cd.origin) @ np.array([-cd.direction[1], cd.direction[0]]))
        m = (perp < 8) & (s_pts > -2) & (s_pts < cd.length + 2)
        for c, name in enumerate(CLASS_NAMES):
            mm = m & (labels == c)
            ax2.scatter(s_pts[mm], xyz[mm, 2], s=1, c=COLORS[c], rasterized=True)
        cv = cd.curve(0.5); ax2.plot(np.linspace(0, cd.length, len(cv)), cv[:, 2], c="k", lw=1.2)
        title = f"Span profile: a={cd.a:.0f} m, sag={cd.sag:.2f} m"
        if clearances:
            title += f", min vegetation clearance {clearances[k]['min_dist']:.2f} m"
        ax2.set_title(title); ax2.set_xlabel("distance along span (m)"); ax2.set_ylabel("z (m)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
