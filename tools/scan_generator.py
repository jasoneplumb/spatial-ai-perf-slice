#!/usr/bin/env python3
"""Deterministic parametric LiDAR scan generator.

Casts a 64-beam top lidar ring (and optionally four short-range lidars) into
a procedural street scene (ground plane plus oriented boxes: vehicles,
pedestrians, cyclists, poles, buildings, optionally the ego vehicle body) and
returns hits as model-ready N x 5 points [x, y, z, intensity, elongation] in a
vehicle-like frame (x forward, y left, z up). Optional second returns model a
beam straddling an object edge.

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

GENERATOR_VERSION = 2


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
    # Optional sensor-model extensions (version 2). All off by default, so version-1 parameters reproduce
    # version-1 output exactly; --preset waymo-like turns them on.
    ego_body: bool = False
    ego_length_m: float = 4.8
    ego_width_m: float = 1.9
    ego_height_m: float = 1.7
    short_range_lidars: int = 0
    sr_beams: int = 200
    sr_azimuth_samples: int = 600
    sr_hfov_deg: float = 180.0
    sr_inclination_min_deg: float = -90.0
    sr_inclination_max_deg: float = 30.0
    sr_max_range_m: float = 20.0
    sr_mount_height_m: float = 0.8
    second_returns: bool = False
    beam_divergence_mrad: float = 3.0
    second_return_min_separation_m: float = 0.5


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
    "ego_body": "Ego vehicle as an occluding box centred under the sensor; returns from it are discarded.",
    "ego_length_m": "Assumption: ego box size.",
    "short_range_lidars": "Published: four short-range lidars (front, side-left, side-right, rear); 0 disables.",
    "sr_beams": "Assumption: commonly cited short-range range-image height; not verified against an official source.",
    "sr_azimuth_samples": "Assumption: commonly cited short-range range-image width; not verified.",
    "sr_hfov_deg": "Assumption: horizontal field of view of each short-range lidar.",
    "sr_inclination_min_deg": "Published: Waymo Open Dataset paper Table 2, F/SL/SR/R VFOV [-90, +30] deg.",
    "sr_inclination_max_deg": "Published: same as sr_inclination_min_deg.",
    "sr_max_range_m": "Published: short-range lidars restricted to 20 m (same table).",
    "sr_mount_height_m": "Assumption: short-range lidar height above ground; mounts sit just outside the ego box.",
    "second_returns": "Published: two returns per pulse are provided. Modeled here only at object edges (see below).",
    "beam_divergence_mrad": "Assumption: offset of the second probe ray that finds an edge's background surface.",
    "second_return_min_separation_m": "Assumption: minimum range gap between first and second return.",
}

PRESETS = {
    # sr_beams x sr_azimuth_samples is calibrated so that the reference scene (default scene parameters,
    # seeds 1 and 7) totals ~183k points, within 4% of the 177K published average points per frame
    # (all lidars, all returns; Waymo Open Dataset paper Table 1). Calibrated, not measured.
    "waymo-like": {"ego_body": True, "short_range_lidars": 4, "second_returns": True,
                   "sr_beams": 100, "sr_azimuth_samples": 300},
}

# Not modeled: vegetation, terrain slope, sensor motion during the sweep,
# second returns other than at object edges, and physical intensity/elongation.


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


INTENSITY = {"ground": 0.08, "vehicle": 0.35, "cyclist": 0.25, "pedestrian": 0.15, "pole": 0.3, "building": 0.2,
             "ego": 0.35}  # ego returns are discarded; entry keeps the lookup total


def _cast(o, d, boxes, ground_z):
    """First hit per ray. o: (3,) or (R, 3) origins; d: (R, 3) unit directions.

    Returns range t (inf on a miss) and surface id: -2 miss, -1 ground, i >= 0 boxes[i].
    """
    t = np.full(d.shape[0], np.inf)
    surf = np.full(d.shape[0], -2, dtype=np.int32)
    oz = o[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 2] < 0, (ground_z - oz) / d[:, 2], np.inf)
    t, surf = np.where(tg < t, tg, t), np.where(tg < t, -1, surf)
    for i, b in enumerate(boxes):
        r_inv = _rot_z(-b["yaw"])
        ol = (o - np.asarray(b["center"])) @ r_inv.T
        dl = d @ r_inv.T
        half = np.asarray(b["size"]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - ol) / dl
            t2 = (half - ol) / dl
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmin > 0)
        closer = hit & (tmin < t)
        t = np.where(closer, tmin, t)
        surf = np.where(closer, i, surf)
    return t, surf


def _returns(o, d, t, surf, keep, rng, p, surf_kind, intensity_scale=1.0):
    """Turn hits into points: dropout, range noise, placeholder intensity/elongation."""
    keep = keep & (rng.random(t.shape[0]) >= p.dropout)
    t = t + rng.normal(0.0, p.range_noise_m, t.shape[0])
    o_k = o[keep] if o.ndim == 2 else o
    xyz = o_k + t[keep, None] * d[keep]
    k = surf_kind[surf[keep] + 2]
    base = np.array([INTENSITY[n] for n in KIND_ORDER])[k] * intensity_scale
    intensity = np.clip(base + rng.normal(0.0, 0.03, k.shape[0]), 0.0, 1.0)
    elongation = np.abs(rng.normal(0.0, 0.05, k.shape[0]))
    return np.column_stack([xyz, intensity, elongation]).astype(np.float32), k


KIND_ORDER = ["ground", "building", "cyclist", "pedestrian", "pole", "vehicle", "ego"]


def _ring(incl_min, incl_max, beams, gamma, az_start, az_span, samples, full_circle):
    u = np.linspace(0.0, 1.0, beams) ** gamma
    incl = np.deg2rad(incl_min + (incl_max - incl_min) * u)
    az = (np.linspace(-np.pi, np.pi, samples, endpoint=False) if full_circle
          else az_start + np.linspace(-az_span / 2, az_span / 2, samples))
    inc_g, az_g = np.meshgrid(incl, az, indexing="ij")  # beam-major, like a range image
    return np.stack([np.cos(inc_g) * np.cos(az_g), np.cos(inc_g) * np.sin(az_g), np.sin(inc_g)], -1).reshape(-1, 3)


def _short_range_mounts(p):
    """(name, origin, yaw) for front, side-left, side-right, rear, just outside the ego box."""
    hx, hy, z = p.ego_length_m / 2 + 0.05, p.ego_width_m / 2 + 0.05, p.ground_z_m + p.sr_mount_height_m
    mounts = [("front", (hx, 0.0, z), 0.0), ("side_left", (0.0, hy, z), np.pi / 2),
              ("side_right", (0.0, -hy, z), -np.pi / 2), ("rear", (-hx, 0.0, z), np.pi)]
    return mounts[:p.short_range_lidars]


def _second_probe(d, rng, divergence):
    """Directions offset from d by `divergence` radians in a random perpendicular direction."""
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u = np.cross(d, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    phi = rng.uniform(0.0, 2 * np.pi, d.shape[0])[:, None]
    d2 = d + divergence * (np.cos(phi) * u + np.sin(phi) * v)
    return d2 / np.linalg.norm(d2, axis=1, keepdims=True)


def generate(p: ScanParams, seed: int):
    rng = np.random.default_rng(seed)
    boxes = build_scene(p, rng)
    ego_index = None
    if p.ego_body:
        ego_index = len(boxes)
        boxes = boxes + [{"kind": "ego", "center": [0.0, 0.0, p.ground_z_m + p.ego_height_m / 2],
                          "size": [p.ego_length_m, p.ego_width_m, p.ego_height_m], "yaw": 0.0}]
    # surface id (-2 miss, -1 ground, i box) + 2 -> index into surf_kind -> index into KIND_ORDER
    surf_kind = np.array([0, 0] + [KIND_ORDER.index(b["kind"]) for b in boxes])

    sensors = [("top", np.array([0.0, 0.0, p.ground_z_m + p.sensor_height_m]),
                _ring(p.inclination_min_deg, p.inclination_max_deg, p.beams, p.inclination_gamma,
                      0.0, 0.0, p.azimuth_samples, True), p.max_range_m)]
    for name, origin, yaw in _short_range_mounts(p):
        sensors.append((name, np.array(origin),
                        _ring(p.sr_inclination_min_deg, p.sr_inclination_max_deg, p.sr_beams, 1.0,
                              yaw, np.deg2rad(p.sr_hfov_deg), p.sr_azimuth_samples, False), p.sr_max_range_m))

    chunks, kinds, per_sensor, rays = [], [], {}, 0
    for name, o, d, max_range in sensors:
        rays += d.shape[0]
        t, surf = _cast(o, d, boxes, p.ground_z_m)
        valid = np.isfinite(t) & (t >= p.min_range_m) & (t <= max_range)
        if ego_index is not None:
            valid &= surf != ego_index  # self-returns are discarded
        pts, k = _returns(o, d, t, surf, valid, rng, p, surf_kind)
        chunks.append(pts)
        kinds.append(k)
        per_sensor[name] = {"rays": int(d.shape[0]), "first_returns": int(pts.shape[0])}
        if p.second_returns:
            # A beam straddling an object edge also returns from the surface behind it: probe a ray offset
            # by the beam divergence and keep its hit when it lands on a different, farther surface.
            d2 = _second_probe(d, rng, p.beam_divergence_mrad * 1e-3)
            t2, surf2 = _cast(o, d2, boxes, p.ground_z_m)
            second = (valid & np.isfinite(t2) & (t2 <= max_range) & (surf2 != surf)
                      & (t2 > t + p.second_return_min_separation_m))
            if ego_index is not None:
                second &= surf2 != ego_index
            pts2, k2 = _returns(o, d2, t2, surf2, second, rng, p, surf_kind, intensity_scale=0.5)
            chunks.append(pts2)
            kinds.append(k2)
            per_sensor[name]["second_returns"] = int(pts2.shape[0])

    points = np.concatenate(chunks, axis=0)
    k = np.concatenate(kinds)
    present = ["ground"] + sorted({b["kind"] for b in boxes if b["kind"] != "ego"})
    counts = {n: int((k == KIND_ORDER.index(n)).sum()) for n in present}
    scene = {
        "boxes": {n: sum(1 for b in boxes if b["kind"] == n) for n in present if n != "ground"},
        "rays": int(rays),
        "returns": int(points.shape[0]),
        "returns_by_surface": counts,
        "per_sensor": per_sensor,
    }
    return points, scene


def _bool(s):
    if s.lower() in ("1", "true", "yes", "on"):
        return True
    if s.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def _add_param_args(ap):
    # Defaults are None so that explicit flags can be told apart from preset values.
    for f in dataclasses.fields(ScanParams):
        kind = _bool if isinstance(f.default, bool) else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, default=None,
                        help=f"{PARAM_NOTES.get(f.name, '')} (default {f.default})")


def params_from_args(args):
    values = dict(PRESETS[args.preset]) if args.preset else {}
    values.update({f.name: getattr(args, f.name) for f in dataclasses.fields(ScanParams)
                   if getattr(args, f.name) is not None})
    return dataclasses.replace(ScanParams(), **values)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="output .npy (N x 5 float32); a .json sidecar is written beside it")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="named parameter set; explicit flags override it")
    _add_param_args(ap)
    args = ap.parse_args()

    params = params_from_args(args)
    points, scene = generate(params, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, points)
    meta = {
        "generator": "tools/scan_generator.py",
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "preset": args.preset,
        "params": dataclasses.asdict(params),
        "param_notes": PARAM_NOTES,
        "not_modeled": ([] if params.short_range_lidars else ["short-range lidars"])
                       + ([] if params.second_returns else ["second returns"])
                       + ([] if params.ego_body else ["ego vehicle body"])
                       + (["second returns other than at object edges"] if params.second_returns else [])
                       + ["vegetation", "terrain slope", "motion during sweep", "physical intensity/elongation"],
        "scene": scene,
        "points_shape": list(points.shape),
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("seed", "scene", "points_shape", "points_sha256")}, indent=2))


if __name__ == "__main__":
    main()
