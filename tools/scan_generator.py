#!/usr/bin/env python3
"""Deterministic parametric LiDAR scan generator.

Casts one ring of beams from a single sensor into a procedural street scene
(ground plane plus oriented boxes: vehicles, pedestrians, cyclists, poles,
buildings) and returns first-return hits as model-ready N x 5 points
[x, y, z, intensity, elongation] in a vehicle-like frame (x forward, y left,
z up).

The geometry is the point: beam rings on the ground, returns concentrated on
surfaces, and occlusion. That structure determines voxel occupancy and
sparse-convolution active sites. Intensity and elongation are placeholder
values with no physical model. No dataset content is used; sensor defaults
are taken from published specifications or marked as assumptions in
PARAM_NOTES.

Same parameters + same seed -> bit-identical output.
"""
import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 1


@dataclass
class ScanParams:
    # Sensor
    beams: int = 64
    azimuth_samples: int = 2650
    inclination_min_deg: float = -17.6
    inclination_max_deg: float = 2.4
    inclination_gamma: float = 1.0
    max_range_m: float = 75.0
    min_range_m: float = 1.0
    sensor_height_m: float = 2.0
    ground_z_m: float = 0.0
    dropout: float = 0.05
    range_noise_m: float = 0.02
    # Scene
    road_half_width_m: float = 7.0
    sidewalk_width_m: float = 3.0
    scene_half_length_m: float = 75.0
    vehicles: int = 30
    pedestrians: int = 20
    cyclists: int = 6
    poles: int = 24
    buildings_per_side: int = 10
    building_gap_m: float = 4.0


PARAM_NOTES = {
    "beams": "Requested design: 64-beam ring.",
    "azimuth_samples": "Assumption: commonly cited top-lidar range-image width; not verified against an official source.",
    "inclination_min_deg": "Published: Waymo Open Dataset paper (arXiv 1912.04838) Table 2, TOP VFOV [-17.6, +2.4] deg.",
    "inclination_max_deg": "Published: same as inclination_min_deg.",
    "inclination_gamma": "Beam spacing warp; 1.0 = uniform in inclination. Real per-beam inclinations are not used.",
    "max_range_m": "Published: TOP range restricted to 75 m (same table).",
    "sensor_height_m": "Assumption: sensor height above ground.",
    "ground_z_m": "Assumption: ground height in the output frame.",
    "dropout": "Assumption: fraction of rays with no return.",
    "range_noise_m": "Assumption: Gaussian range noise sigma.",
}

# Not modeled: the four short-range lidars, second returns, the ego vehicle
# body, vegetation, terrain slope, sensor motion during the sweep.


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _place(rng, n, x_range, y_choices, size_range, yaw_fn, taken, clearance, kind, ground_z):
    """Rejection-sample non-overlapping boxes in BEV."""
    boxes = []
    attempts = 0
    while len(boxes) < n and attempts < 200 * max(n, 1):
        attempts += 1
        lo, hi = np.asarray(size_range[0]), np.asarray(size_range[1])
        l, w, h = rng.uniform(lo, hi)
        x = rng.uniform(*x_range)
        y = y_choices(rng)
        r = 0.5 * np.hypot(l, w)
        if abs(x) < 6.0 and abs(y) < 2.5:  # keep the ego footprint clear
            continue
        if any(np.hypot(x - tx, y - ty) < r + tr + clearance for tx, ty, tr in taken):
            continue
        taken.append((x, y, r))
        boxes.append({"kind": kind, "center": [x, y, ground_z + h / 2], "size": [l, w, h], "yaw": yaw_fn(rng)})
    return boxes


def build_scene(p: ScanParams, rng):
    L, rw, sw = p.scene_half_length_m, p.road_half_width_m, p.sidewalk_width_m
    g = p.ground_z_m
    taken = []
    lanes = [-rw * 0.75, -rw * 0.25, rw * 0.25, rw * 0.75]

    def lane(r):
        return lanes[r.integers(len(lanes))] + r.normal(0.0, 0.2)

    def sidewalk(r):
        return r.choice([-1.0, 1.0]) * (rw + r.uniform(0.3, sw - 0.3))

    def road_edge(r):
        return r.choice([-1.0, 1.0]) * (rw - 0.6)

    def heading(r):
        return r.choice([0.0, np.pi]) + r.normal(0.0, 0.05)

    def any_yaw(r):
        return r.uniform(-np.pi, np.pi)

    boxes = []
    boxes += _place(rng, p.vehicles, (-L, L), lane, ((3.8, 1.7, 1.4), (5.2, 2.1, 2.0)), heading, taken, 1.0, "vehicle", g)
    boxes += _place(rng, p.cyclists, (-L, L), road_edge, ((1.6, 0.5, 1.6), (1.9, 0.8, 1.8)), heading, taken, 0.5, "cyclist", g)
    boxes += _place(rng, p.pedestrians, (-L, L), sidewalk, ((0.5, 0.5, 1.5), (0.9, 0.9, 1.9)), any_yaw, taken, 0.3, "pedestrian", g)
    boxes += _place(rng, p.poles, (-L, L), lambda r: r.choice([-1.0, 1.0]) * (rw + 0.4),
                    ((0.25, 0.25, 4.0), (0.4, 0.4, 8.0)), lambda r: 0.0, taken, 0.2, "pole", g)

    # Buildings: contiguous blocks with gaps behind each sidewalk.
    setback = rw + sw
    for side in (-1.0, 1.0):
        edges = np.sort(rng.uniform(-L, L, max(p.buildings_per_side - 1, 0)))
        bounds = np.concatenate([[-L], edges, [L]])
        for x0, x1 in zip(bounds[:-1], bounds[1:]):
            length = x1 - x0 - p.building_gap_m
            if length < 2.0:
                continue
            depth = rng.uniform(10.0, 20.0)
            height = rng.uniform(6.0, 20.0)
            boxes.append({"kind": "building", "center": [(x0 + x1) / 2, side * (setback + depth / 2), g + height / 2],
                          "size": [length, depth, height], "yaw": 0.0})
    return boxes


INTENSITY = {"ground": 0.08, "vehicle": 0.35, "cyclist": 0.25, "pedestrian": 0.15, "pole": 0.3, "building": 0.2}


def generate(p: ScanParams, seed: int):
    rng = np.random.default_rng(seed)
    boxes = build_scene(p, rng)

    u = np.linspace(0.0, 1.0, p.beams) ** p.inclination_gamma
    incl = np.deg2rad(p.inclination_min_deg + (p.inclination_max_deg - p.inclination_min_deg) * u)
    az = np.linspace(-np.pi, np.pi, p.azimuth_samples, endpoint=False)
    inc_g, az_g = np.meshgrid(incl, az, indexing="ij")  # beam-major, like a range image
    d = np.stack([np.cos(inc_g) * np.cos(az_g), np.cos(inc_g) * np.sin(az_g), np.sin(inc_g)], -1).reshape(-1, 3)
    o = np.array([0.0, 0.0, p.ground_z_m + p.sensor_height_m])

    t = np.full(d.shape[0], np.inf)
    kind = np.full(d.shape[0], -1, dtype=np.int32)
    kinds = ["ground"] + sorted({b["kind"] for b in boxes})
    kind_id = {k: i for i, k in enumerate(kinds)}

    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 2] < 0, (p.ground_z_m - o[2]) / d[:, 2], np.inf)
    t, kind = np.where(tg < t, tg, t), np.where(tg < t, kind_id["ground"], kind)

    for b in boxes:
        r_inv = _rot_z(-b["yaw"])
        ol = r_inv @ (o - np.asarray(b["center"]))
        dl = d @ r_inv.T
        half = np.asarray(b["size"]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - ol) / dl
            t2 = (half - ol) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmin > 0)
        closer = hit & (tmin < t)
        t = np.where(closer, tmin, t)
        kind = np.where(closer, kind_id[b["kind"]], kind)

    keep = np.isfinite(t) & (t >= p.min_range_m) & (t <= p.max_range_m)
    keep &= rng.random(t.shape[0]) >= p.dropout
    t = t + rng.normal(0.0, p.range_noise_m, t.shape[0])
    xyz = o + t[keep, None] * d[keep]
    k = kind[keep]

    base = np.array([INTENSITY[n] for n in kinds])[k]
    intensity = np.clip(base + rng.normal(0.0, 0.03, k.shape[0]), 0.0, 1.0)
    elongation = np.abs(rng.normal(0.0, 0.05, k.shape[0]))
    points = np.column_stack([xyz, intensity, elongation]).astype(np.float32)

    counts = {n: int((k == i).sum()) for n, i in kind_id.items()}
    scene = {
        "boxes": {n: sum(1 for b in boxes if b["kind"] == n) for n in kinds if n != "ground"},
        "rays": int(d.shape[0]),
        "returns": int(points.shape[0]),
        "returns_by_surface": counts,
    }
    return points, scene


def _add_param_args(ap):
    for f in dataclasses.fields(ScanParams):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default,
                        help=PARAM_NOTES.get(f.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="output .npy (N x 5 float32); a .json sidecar is written beside it")
    _add_param_args(ap)
    args = ap.parse_args()

    params = ScanParams(**{f.name: getattr(args, f.name) for f in dataclasses.fields(ScanParams)})
    points, scene = generate(params, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, points)
    meta = {
        "generator": "tools/scan_generator.py",
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "params": dataclasses.asdict(params),
        "param_notes": PARAM_NOTES,
        "not_modeled": ["short-range lidars", "second returns", "ego vehicle body", "vegetation",
                        "terrain slope", "motion during sweep", "physical intensity/elongation"],
        "scene": scene,
        "points_shape": list(points.shape),
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("seed", "scene", "points_shape", "points_sha256")}, indent=2))


if __name__ == "__main__":
    main()
