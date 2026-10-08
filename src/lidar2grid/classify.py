"""Point classifier with spatially blocked evaluation.

Neighbouring points are strongly correlated, so a random train/test split leaks
information and inflates scores. We split by spatial blocks instead.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import CLASS_NAMES


def spatial_blocks(xyz, block=50.0):
    bx = np.floor(xyz[:, 0] / block).astype(np.int64)
    by = np.floor(xyz[:, 1] / block).astype(np.int64)
    return bx * 100_003 + by


def make_model(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
        class_weight="balanced", early_stopping=True, random_state=seed,
    )


def per_class_metrics(y_true, y_pred, n_classes=len(CLASS_NAMES)):
    rows = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c)); fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(cls=CLASS_NAMES[c], support=int(np.sum(y_true == c)), precision=p, recall=r,
                         f1=2 * p * r / (p + r) if p + r else 0.0, iou=tp / (tp + fp + fn)))
    return rows


def spatial_cv(X, y, groups, n_splits=4, seed=0):
    """Out-of-fold predictions using spatial block folds."""
    oof = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model(seed).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def format_metrics(rows):
    lines = ["| class | support | precision | recall | F1 | IoU |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['cls']} | {r['support']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['iou']:.3f} |")
    miou = np.mean([r["iou"] for r in rows])
    lines.append(f"\nmIoU: **{miou:.3f}**")
    return "\n".join(lines)
