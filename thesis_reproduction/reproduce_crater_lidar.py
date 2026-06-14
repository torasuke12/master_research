#!/usr/bin/env python3
"""Reproduce the crater/LiDAR simulation in Zhou et al. (Measurement, 2023).

The original paper uses PANGU/TIN terrain and proprietary implementation
details.  This script keeps the reproducible core: analytic crater terrain,
LiDAR ray sampling, DBSCAN crater detection, RANSAC circle completion, and the
paper's DP/CR/terrain-frequency metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_NEGATIVE_OBSTACLE_THRESHOLD = -0.25


@dataclass(frozen=True)
class Crater:
    x: float
    y: float
    radius: float
    depth: float
    rim_height: float


@dataclass(frozen=True)
class Rock:
    x: float
    y: float
    diameter: float
    height: float


@dataclass(frozen=True)
class LidarConfig:
    origin: tuple[float, float, float] = (0.0, -10.0, 1.5)
    vertical_fov_deg: float = 16.0
    horizontal_fov_deg: float = 60.0
    vertical_pixels: int = 32
    horizontal_pixels: int = 128
    pitch_center_deg: float = -6.0
    max_range: float = 50.0
    range_sigma: float = 0.0
    horizontal_sigma_deg: float = 0.0
    vertical_sigma_deg: float = 0.0


def rim_height_ratio_from_ddr(ddr: float) -> float:
    """Return rim-height/diameter ratio from Table 1 in Zhou et al.

    Table 1 reports ranges by morphology:
    mature DDR 0.11-0.13, rim/diameter 0.008-0.030;
    young DDR 0.17-0.19, rim/diameter 0.016-0.045;
    fresh DDR 0.23-0.25, rim/diameter 0.022-0.060.
    The simulation uses no raised rim below DDR 0.11 because Table 1 does not
    specify rim heights there.  For DDR 0.11 and above, it uses the midpoint of
    each reported range and interpolates by DDR.
    """
    anchors = [(0.12, 0.019), (0.18, 0.0305), (0.24, 0.041)]
    if ddr < 0.11:
        return 0.0
    if ddr <= anchors[0][0]:
        return anchors[0][1]
    if ddr >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= ddr <= x1:
            t = (ddr - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return anchors[1][1]


def crater_from_ddr(x: float, y: float, radius: float, ddr: float, rim_ratio: float | None = None) -> Crater:
    diameter = 2.0 * radius
    if rim_ratio is None:
        rim_ratio = rim_height_ratio_from_ddr(ddr)
    return Crater(
        x=x,
        y=y,
        radius=radius,
        depth=ddr * diameter,
        rim_height=rim_ratio * diameter,
    )


def terrain_height(x: np.ndarray, y: np.ndarray, craters: Iterable[Crater], rocks: Iterable[Rock] = ()) -> np.ndarray:
    z = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)
    x = np.asarray(x)
    y = np.asarray(y)

    for crater in craters:
        dx = x - crater.x
        dy = y - crater.y
        rho = np.sqrt(dx * dx + dy * dy)
        inside = rho <= crater.radius
        crater_z = (rho * rho) * (crater.depth + crater.rim_height) / crater.radius**2 - crater.depth
        z = np.where(inside, np.minimum(z, crater_z), z)

    for rock in rocks:
        dx = x - rock.x
        dy = y - rock.y
        rho2 = dx * dx + dy * dy
        mask = rho2 <= (rock.diameter / 2.0) ** 2
        rock_z = rock.height - (4.0 * rock.height * rho2 / rock.diameter**2)
        z = np.where(mask, np.maximum(z, rock_z), z)

    return z


def terrain_frequency(z: np.ndarray, width_x: float, width_y: float) -> float:
    spectrum = np.abs(np.fft.fft2(z))
    nx, ny = z.shape
    u = np.fft.fftfreq(nx, d=width_x / nx)
    v = np.fft.fftfreq(ny, d=width_y / ny)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = np.sqrt(uu * uu + vv * vv)
    denom = spectrum.sum() * math.sqrt(width_x * width_x + width_y * width_y)
    if denom == 0:
        return 0.0
    return float((spectrum * radial).sum() / denom)


def ray_sample_terrain(
    craters: list[Crater],
    rocks: list[Rock],
    cfg: LidarConfig,
    rng: np.random.Generator,
    step: float = 0.18,
) -> np.ndarray:
    ox, oy, oz = cfg.origin
    h_angles = np.linspace(-cfg.horizontal_fov_deg / 2, cfg.horizontal_fov_deg / 2, cfg.horizontal_pixels)
    v_angles = np.linspace(
        cfg.pitch_center_deg - cfg.vertical_fov_deg / 2,
        cfg.pitch_center_deg + cfg.vertical_fov_deg / 2,
        cfg.vertical_pixels,
    )
    points: list[tuple[float, float, float]] = []
    ranges = np.arange(step, cfg.max_range + step, step)

    for v_deg in v_angles:
        for h_deg in h_angles:
            h = math.radians(h_deg + rng.normal(0.0, cfg.horizontal_sigma_deg))
            v = math.radians(v_deg + rng.normal(0.0, cfg.vertical_sigma_deg))
            direction = np.array([math.cos(v) * math.sin(h), math.cos(v) * math.cos(h), math.sin(v)])
            xs = ox + ranges * direction[0]
            ys = oy + ranges * direction[1]
            zs = oz + ranges * direction[2]
            ground = terrain_height(xs, ys, craters, rocks)
            hit = np.nonzero(zs <= ground)[0]
            if len(hit) == 0:
                continue
            k = int(hit[0])
            lo = ranges[max(0, k - 1)]
            hi = ranges[k]
            for _ in range(5):
                mid = (lo + hi) / 2.0
                mx = ox + mid * direction[0]
                my = oy + mid * direction[1]
                mz = oz + mid * direction[2]
                if mz <= terrain_height(mx, my, craters, rocks):
                    hi = mid
                else:
                    lo = mid
            measured = max(0.0, hi + rng.normal(0.0, cfg.range_sigma))
            px, py, pz = np.array([ox, oy, oz]) + measured * direction
            points.append((float(px), float(py), float(pz)))
    return np.asarray(points)


def dbscan(points_xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(points_xy) == 0:
        return np.empty(0, dtype=int)
    tree = cKDTree(points_xy)
    labels = np.full(len(points_xy), -1, dtype=int)
    visited = np.zeros(len(points_xy), dtype=bool)
    cluster_id = 0
    for idx in range(len(points_xy)):
        if visited[idx]:
            continue
        visited[idx] = True
        neighbors = tree.query_ball_point(points_xy[idx], eps)
        if len(neighbors) < min_samples:
            continue
        labels[idx] = cluster_id
        seeds = list(neighbors)
        while seeds:
            current = seeds.pop()
            if not visited[current]:
                visited[current] = True
                current_neighbors = tree.query_ball_point(points_xy[current], eps)
                if len(current_neighbors) >= min_samples:
                    seeds.extend(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1
    return labels


def circle_from_3_points(sample: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = sample
    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 1e-9:
        return None
    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / det
    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / det
    radius = math.hypot(x1 - ux, y1 - uy)
    return float(ux), float(uy), float(radius)


def ransac_circle(points_xy: np.ndarray, rng: np.random.Generator, iterations: int = 300, threshold: float = 0.35) -> tuple[float, float, float] | None:
    if len(points_xy) < 3:
        return None
    best_model = None
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        sample = points_xy[rng.choice(len(points_xy), size=3, replace=False)]
        model = circle_from_3_points(sample)
        if model is None:
            continue
        cx, cy, radius = model
        if not 0.5 <= radius <= 30.0:
            continue
        residual = np.abs(np.linalg.norm(points_xy - np.array([cx, cy]), axis=1) - radius)
        inliers = residual < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_model = model
            best_inliers = inliers
    if best_model is None or best_inliers is None or best_inliers.sum() < 3:
        return None
    inlier_points = points_xy[best_inliers]
    x = inlier_points[:, 0]
    y = inlier_points[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = math.sqrt(max(0.0, c + cx * cx + cy * cy))
    return float(cx), float(cy), float(radius)


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    part1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    part2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    part3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return part1 + part2 - part3


def identify_craters(
    points: np.ndarray,
    craters: list[Crater],
    rng: np.random.Generator,
    negative_threshold: float = DEFAULT_NEGATIVE_OBSTACLE_THRESHOLD,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    if len(points) == 0:
        return 0.0, 0.0, []
    negative = points[points[:, 2] < negative_threshold]
    labels = dbscan(negative[:, :2], eps=1.2, min_samples=4)
    detections: list[tuple[float, float, float]] = []
    for label in sorted(set(labels) - {-1}):
        cluster = negative[labels == label, :2]
        model = ransac_circle(cluster, rng)
        if model is not None:
            detections.append(model)

    detected = 0
    cr_values = []
    for crater in craters:
        true_radius = crater.radius
        best_cr = 0.0
        for cx, cy, radius in detections:
            dist = math.hypot(cx - crater.x, cy - crater.y)
            if dist < max(radius, true_radius):
                area = circle_intersection_area(true_radius, radius, dist)
                best_cr = max(best_cr, area / (math.pi * true_radius * true_radius))
        if best_cr > 0.05:
            detected += 1
            cr_values.append(best_cr)
    dp = detected / len(craters) if craters else 0.0
    cr = float(np.mean(cr_values)) if cr_values else 0.0
    return dp, cr, detections


def single_crater_trial(
    *,
    radius: float = 5.0,
    ddr: float = 0.17,
    lidar: LidarConfig | None = None,
    seed: int = 0,
    negative_threshold: float = DEFAULT_NEGATIVE_OBSTACLE_THRESHOLD,
) -> tuple[float, float, int, list[tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    crater = crater_from_ddr(0.0, 15.0, radius, ddr, rim_ratio=0.0)
    cfg = lidar or LidarConfig()
    points = ray_sample_terrain([crater], [], cfg, rng)
    dp, cr, detections = identify_craters(points, [crater], rng, negative_threshold)
    return dp, cr, len(points), detections


def terrain_frequency_experiment(out_dir: Path, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    xs = np.linspace(-50, 50, 241)
    ys = np.linspace(-50, 50, 241)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    for radius_lo, radius_hi in [(3.0, 5.0), (5.5, 7.5), (8.0, 10.0), (10.5, 12.5)]:
        for count in range(2, 18):
            craters = []
            for _ in range(count):
                radius = rng.uniform(radius_lo, radius_hi)
                ddr = rng.choice([0.12, 0.18, 0.24])
                craters.append(crater_from_ddr(rng.uniform(-42, 42), rng.uniform(-42, 42), radius, ddr))
            z = terrain_height(xx, yy, craters)
            tf = terrain_frequency(z, 100.0, 100.0)
            volume = sum(math.pi * c.radius * c.radius * c.depth for c in craters)
            rows.append({"radius_mid": (radius_lo + radius_hi) / 2, "count": count, "volume": volume, "terrain_frequency": tf})

    fig, ax = plt.subplots(figsize=(6, 4))
    for radius_mid in sorted(set(row["radius_mid"] for row in rows)):
        group = [row for row in rows if row["radius_mid"] == radius_mid]
        ax.scatter([row["volume"] for row in group], [row["terrain_frequency"] for row in group], s=22, label=f"R~{radius_mid:.1f} m")
    ax.set_xlabel("Volume of craters (m^3)")
    ax.set_ylabel("Terrain frequency (1/m)")
    ax.set_title("Terrain frequency vs crater volume")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "terrain_frequency_vs_volume.png", dpi=160)
    plt.close(fig)
    return rows


def density_experiment(out_dir: Path, seed: int, negative_threshold: float) -> list[dict[str, float]]:
    rows = []
    configs = [(12, 48), (16, 64), (24, 96), (32, 128), (40, 128)]
    for vp, hp in configs:
        cfg = LidarConfig(vertical_pixels=vp, horizontal_pixels=hp)
        dps, crs, densities = [], [], []
        for trial in range(5):
            dp, cr, n_points, _ = single_crater_trial(lidar=cfg, seed=seed + trial + vp + hp, negative_threshold=negative_threshold)
            dps.append(dp)
            crs.append(cr)
            densities.append(n_points / (cfg.max_range * 2.0 * cfg.max_range * math.tan(math.radians(cfg.horizontal_fov_deg / 2.0))))
        rows.append({"vertical_pixels": vp, "horizontal_pixels": hp, "density": float(np.mean(densities)), "dp": float(np.mean(dps)), "cr": float(np.mean(crs))})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["density"] for row in rows], [row["dp"] for row in rows], "o-")
    ax.set_xlabel("Point density proxy (pts/m^2)")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("DP vs point density")
    fig.tight_layout()
    fig.savefig(out_dir / "dp_vs_point_density.png", dpi=160)
    plt.close(fig)
    return rows


def error_experiment(out_dir: Path, seed: int, negative_threshold: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    range_rows = []
    for sigma in np.linspace(0.0, 0.12, 7):
        dps, crs = [], []
        for trial in range(6):
            cfg = LidarConfig(range_sigma=float(sigma), horizontal_fov_deg=90.0, horizontal_pixels=128)
            dp, cr, _, _ = single_crater_trial(lidar=cfg, seed=seed + 1000 + trial, negative_threshold=negative_threshold)
            dps.append(dp)
            crs.append(cr)
        range_rows.append({"range_sigma_m": float(sigma), "dp": float(np.mean(dps)), "cr": float(np.mean(crs))})

    angle_rows = []
    for sigma in np.linspace(0.0, 0.5, 6):
        dps, crs = [], []
        for trial in range(6):
            cfg = LidarConfig(vertical_sigma_deg=float(sigma), horizontal_fov_deg=90.0, horizontal_pixels=128)
            dp, cr, _, _ = single_crater_trial(lidar=cfg, seed=seed + 2000 + trial, negative_threshold=negative_threshold)
            dps.append(dp)
            crs.append(cr)
        angle_rows.append({"vertical_sigma_deg": float(sigma), "dp": float(np.mean(dps)), "cr": float(np.mean(crs))})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["range_sigma_m"] for row in range_rows], [row["dp"] for row in range_rows], "o-", label="range error")
    ax.plot([row["vertical_sigma_deg"] for row in angle_rows], [row["dp"] for row in angle_rows], "s-", label="vertical angle error")
    ax.set_xlabel("Error sigma (m for range, deg for angle)")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("DP degradation under LiDAR errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dp_vs_errors.png", dpi=160)
    plt.close(fig)
    return range_rows, angle_rows


def ddr_experiment(out_dir: Path, seed: int, negative_threshold: float) -> list[dict[str, float]]:
    rows = []
    cfg = LidarConfig(horizontal_fov_deg=90.0, horizontal_pixels=128)
    for ddr in [0.05, 0.10, 0.15, 0.20, 0.25]:
        crs, dps = [], []
        for trial in range(5):
            dp, cr, _, _ = single_crater_trial(ddr=ddr, lidar=cfg, seed=seed + 3000 + trial, negative_threshold=negative_threshold)
            crs.append(cr)
            dps.append(dp)
        rows.append({"ddr": ddr, "dp": float(np.mean(dps)), "cr": float(np.mean(crs))})

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["ddr"] for row in rows], [row["cr"] for row in rows], "o-")
    ax.set_xlabel("Depth to diameter ratio (DDR)")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CR vs crater DDR")
    fig.tight_layout()
    fig.savefig(out_dir / "cr_vs_ddr.png", dpi=160)
    plt.close(fig)
    return rows


def visual_demo(out_dir: Path, seed: int, negative_threshold: float) -> None:
    rng = np.random.default_rng(seed)
    crater = crater_from_ddr(0.0, 15.0, 5.0, 0.17, rim_ratio=0.0)
    rock = Rock(x=3.0, y=16.0, diameter=1.0, height=0.45)
    cfg = LidarConfig(horizontal_fov_deg=90.0, horizontal_pixels=128)
    points = ray_sample_terrain([crater], [rock], cfg, rng)
    _, _, detections = identify_craters(points, [crater], rng, negative_threshold)

    xs = np.linspace(-8, 8, 180)
    ys = np.linspace(7, 23, 180)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = terrain_height(xx, yy, [crater], [rock])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, zz, levels=30, cmap="viridis")
    negative = points[points[:, 2] < negative_threshold]
    ax.scatter(negative[:, 0], negative[:, 1], s=8, c="black", label="negative LiDAR points")
    actual = plt.Circle((crater.x, crater.y), crater.radius, color="red", fill=False, linewidth=2, label="actual crater")
    ax.add_patch(actual)
    for i, (cx, cy, radius) in enumerate(detections):
        ax.add_patch(plt.Circle((cx, cy), radius, color="deepskyblue", fill=False, linewidth=2, label="completed crater" if i == 0 else None))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("LiDAR partial scan and RANSAC crater completion")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "crater_completion_demo.png", dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/reproduction", help="directory for figures and CSV files")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--negative-threshold", type=float, default=DEFAULT_NEGATIVE_OBSTACLE_THRESHOLD, help="z threshold for negative obstacle points")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_rows = terrain_frequency_experiment(out_dir, args.seed)
    density_rows = density_experiment(out_dir, args.seed, args.negative_threshold)
    range_rows, angle_rows = error_experiment(out_dir, args.seed, args.negative_threshold)
    ddr_rows = ddr_experiment(out_dir, args.seed, args.negative_threshold)
    visual_demo(out_dir, args.seed, args.negative_threshold)

    write_csv(out_dir / "terrain_frequency_vs_volume.csv", tf_rows)
    write_csv(out_dir / "dp_vs_point_density.csv", density_rows)
    write_csv(out_dir / "dp_vs_range_error.csv", range_rows)
    write_csv(out_dir / "dp_vs_vertical_angle_error.csv", angle_rows)
    write_csv(out_dir / "cr_vs_ddr.csv", ddr_rows)

    print(f"Wrote reproduction outputs to {out_dir}")
    print("Key results:")
    print(f"  Max CR in DDR sweep: {max(row['cr'] for row in ddr_rows):.3f}")
    print(f"  DP at largest range error: {range_rows[-1]['dp']:.3f}")
    print(f"  DP at largest vertical angle error: {angle_rows[-1]['dp']:.3f}")


if __name__ == "__main__":
    main()
