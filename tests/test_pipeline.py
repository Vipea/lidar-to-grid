import numpy as np
import pytest

from lidar2grid import GROUND, POLE, VEGETATION, WIRE
from lidar2grid.gis import compare, perturb_gis
from lidar2grid.ground import classify_ground
from lidar2grid.network import build_network, detect_poles, extract_conductors, vegetation_clearance
from lidar2grid.synthetic import make_scene


@pytest.fixture(scope="module")
def scene():
    sc = make_scene(seed=3)
    _, hag = classify_ground(sc.xyz)
    return sc, hag


def test_ground_filter_separates_ground(scene):
    sc, hag = scene
    is_ground = np.abs(hag) < 0.3
    gt = sc.labels == GROUND
    assert (is_ground & gt).sum() / gt.sum() > 0.97
    assert (is_ground & ~gt).sum() / is_ground.sum() < 0.02


def test_network_from_true_labels(scene):
    sc, hag = scene
    y = sc.labels
    poles = detect_poles(sc.xyz[y == POLE], hag[y == POLE])
    assert len(poles) == len(sc.poles)
    cond = extract_conductors(sc.xyz[y == WIRE], poles)
    net = build_network(poles, cond)
    assert len(net["edges"]) == len(sc.poles) - 1
    assert all(e["n_conductors"] == 3 for e in net["edges"])
    assert net["unanchored_conductors"] == 0
    # catenary parameter and sag are recovered: sag ~ L^2 / (8a)
    a = np.median([c.a for c in cond])
    assert abs(a - sc.catenary_a) / sc.catenary_a < 0.1
    span = np.median([c.length for c in cond])
    assert np.median([c.sag for c in cond]) == pytest.approx(span**2 / (8 * sc.catenary_a), rel=0.15)


def test_vegetation_clearance_finds_close_trees(scene):
    sc, hag = scene
    y = sc.labels
    cond = extract_conductors(sc.xyz[y == WIRE], detect_poles(sc.xyz[y == POLE], hag[y == POLE]))
    clr = vegetation_clearance(cond, sc.xyz[y == VEGETATION])
    assert min(c["min_dist"] for c in clr) < 5.0


def test_gis_comparison_flags_errors():
    rng = np.random.default_rng(0)
    sc = make_scene(seed=4)
    gis_poles, gis_lines, bad, dropped = perturb_gis(sc.poles, rng)
    spans = [[sc.poles[k], sc.poles[k + 1]] for k in range(len(sc.poles) - 1)]
    rep = compare(spans, gis_lines, sc.poles, gis_poles)
    assert dropped in rep["missing_from_gis"]
    assert set(bad) <= set(rep["gis_poles_misplaced"])
