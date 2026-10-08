"""End-to-end demo on a synthetic scene: ground -> features -> classifier -> network."""
import argparse, json, time
from pathlib import Path

import numpy as np

from lidar2grid import POLE, VEGETATION, WIRE
from lidar2grid.classify import format_metrics, make_model, per_class_metrics, spatial_blocks, spatial_cv
from lidar2grid.features import compute_features
from lidar2grid.ground import classify_ground
from lidar2grid.network import build_network, detect_poles, extract_conductors, to_geojson, vegetation_clearance
from lidar2grid.synthetic import make_scene
from lidar2grid.gis import compare, perturb_gis
from lidar2grid.viz import plot_scene

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/synthetic"); args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t = time.time()
train, test = make_scene(seed=1), make_scene(seed=2)

def featurise(sc):
    is_ground, hag = classify_ground(sc.xyz)
    X, names = compute_features(sc.xyz, hag)
    return X, hag, is_ground, names

Xtr, _, _, names = featurise(train)
Xte, hag_te, g_te, _ = featurise(test)
print(f"features {Xtr.shape} in {time.time()-t:.1f}s")
oof = spatial_cv(Xtr, train.labels, spatial_blocks(train.xyz))
print("spatial CV (train scene)\n" + format_metrics(per_class_metrics(train.labels, oof)))
model = make_model().fit(Xtr, train.labels)
pred = model.predict(Xte)
print("held-out scene\n" + format_metrics(per_class_metrics(test.labels, pred)))

poles = detect_poles(test.xyz[pred == POLE], hag_te[pred == POLE])
cond = extract_conductors(test.xyz[pred == WIRE], poles)
net = build_network(poles, cond)
clr = vegetation_clearance(cond, test.xyz[pred == VEGETATION])
print(f"poles found {len(poles)} / true {len(test.poles)}; conductors {len(cond)}; spans {len(net['edges'])}")
for e in net["edges"]:
    print(e)
print("catenary a (true %.0f):" % test.catenary_a, np.round([c.a for c in cond], 0))
print("min veg clearance per conductor:", np.round([c['min_dist'] for c in clr], 2))
json.dump(to_geojson(net, cond, clr), open(out / "network.geojson", "w"))

# Compare against an imperfect GIS record of the same line
gis_poles, gis_lines, bad, dropped = perturb_gis(test.poles, np.random.default_rng(0))
P = {p.id: p.xy for p in poles}
spans = [[P[e["u"]], P[e["v"]]] for e in net["edges"]]
rep = compare(spans, gis_lines, [p.xy for p in poles], gis_poles)
print("GIS QA:", rep, "| injected: misplaced poles", bad, "dropped span", dropped)
plot_scene(test.xyz, pred, cond, poles, gis_poles, out / "scene.png", clr)
