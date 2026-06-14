#!/usr/bin/env python3
"""Plot DDR vs CR using the current interactive simulator detection process."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def angle_samples(fov_deg: float, resolution_deg: float) -> np.ndarray:
    count = max(1, math.floor(fov_deg / resolution_deg) + 1)
    if count == 1:
        return np.array([0.0])
    span = resolution_deg * (count - 1)
    start = -span / 2.0
    return np.deg2rad(start + resolution_deg * np.arange(count))


def crater_height(x: np.ndarray | float, y: np.ndarray | float, diameter: float, ddr: float) -> np.ndarray | float:
    radius = diameter / 2.0
    depth = ddr * diameter
    rho = np.hypot(x, y)
    inside = rho <= radius
    z = np.zeros_like(rho, dtype=float) if isinstance(rho, np.ndarray) else 0.0
    crater_z = (rho * rho) * depth / radius**2 - depth
    return np.where(inside, crater_z, z) if isinstance(rho, np.ndarray) else (float(crater_z) if inside else 0.0)


def ray_ground_hit(diameter: float, ddr: float, sensor: tuple[float, float, float], h: float, v: float, max_range: float) -> float | None:
    sx, sy, sz = sensor
    step = 0.12
    previous = 0.5
    r = previous
    while r <= max_range:
        x = sx + r * math.cos(v) * math.sin(h)
        y = sy + r * math.cos(v) * math.cos(h)
        z = sz + r * math.sin(v)
        if z <= crater_height(x, y, diameter, ddr):
            lo, hi = previous, r
            for _ in range(8):
                mid = (lo + hi) / 2.0
                mx = sx + mid * math.cos(v) * math.sin(h)
                my = sy + mid * math.cos(v) * math.cos(h)
                mz = sz + mid * math.sin(v)
                if mz <= crater_height(mx, my, diameter, ddr):
                    hi = mid
                else:
                    lo = mid
            return hi
        previous = r
        r += step
    return None


def simulate_lidar(
    *,
    diameter: float,
    ddr: float,
    front_rim_distance: float,
    sensor_height: float,
    horizontal_fov: float,
    vertical_fov: float,
    angular_resolution: float,
    range_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radius = diameter / 2.0
    sensor = (0.0, -(front_rim_distance + radius), sensor_height)
    h_samples = angle_samples(horizontal_fov, angular_resolution)
    v_samples = angle_samples(vertical_fov, angular_resolution)
    pitch_center = math.atan2(-sensor_height, front_rim_distance + radius)
    max_range = min(75.0, front_rim_distance + diameter * 2.0 + 8.0)
    points: list[tuple[float, float, float]] = []
    ray_index = 0
    sx, sy, sz = sensor

    for v_offset in v_samples:
        v = pitch_center + float(v_offset)
        for h in h_samples:
            hit_range = ray_ground_hit(diameter, ddr, sensor, float(h), v, max_range)
            if hit_range is not None:
                measured = max(0.0, hit_range + rng.normal(0.0, range_noise))
                x = sx + measured * math.cos(v) * math.sin(float(h))
                y = sy + measured * math.cos(v) * math.cos(float(h))
                z = sz + measured * math.sin(v)
                points.append((x, y, z))
            ray_index += 1
    return np.asarray(points, dtype=float)


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
    if abs(det) < 1e-8:
        return None
    a = x1 * x1 + y1 * y1
    b = x2 * x2 + y2 * y2
    c = x3 * x3 + y3 * y3
    cx = (a * (y2 - y3) + b * (y3 - y1) + c * (y1 - y2)) / det
    cy = (a * (x3 - x2) + b * (x1 - x3) + c * (x2 - x1)) / det
    radius = math.hypot(x1 - cx, y1 - cy)
    return float(cx), float(cy), float(radius)


def angular_span(points: np.ndarray, circle: tuple[float, float, float]) -> float:
    cx, cy, _ = circle
    angles = np.sort(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(angles)
    wrap_gap = angles[0] + 2.0 * math.pi - angles[-1]
    return float(2.0 * math.pi - max(float(gaps.max(initial=0.0)), float(wrap_gap)))


def least_squares_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = math.sqrt(max(0.0, c + cx * cx + cy * cy))
    if not math.isfinite(radius):
        return None
    return float(cx), float(cy), float(radius)


def ransac_circle(points: np.ndarray, diameter: float, range_noise: float, rng: np.random.Generator) -> tuple[float, float, float] | None:
    if len(points) < 4:
        return None
    residual_threshold = max(0.18, diameter * 0.035 + range_noise * 2.0)
    min_radius = max(0.7, diameter * 0.18)
    max_radius = max(3.0, diameter * 1.25)
    iterations = min(900, max(180, len(points) * len(points) * 2))
    best_model = None
    best_inliers = None
    best_score = -math.inf

    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        model = circle_from_3_points(sample)
        if model is None:
            continue
        cx, cy, radius = model
        if radius < min_radius or radius > max_radius:
            continue
        residual = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        inliers = residual <= residual_threshold
        if inliers.sum() < 4:
            continue
        mean_residual = float(residual[inliers].mean())
        span = angular_span(points[inliers], model)
        score = int(inliers.sum()) * 20.0 + span * 8.0 - mean_residual * 100.0
        if score > best_score:
            best_score = score
            best_model = model
            best_inliers = inliers

    if best_model is None or best_inliers is None:
        return None
    refined = least_squares_circle(points[best_inliers])
    if refined is None:
        return best_model
    if refined[2] < min_radius or refined[2] > max_radius:
        return best_model
    return refined


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))
    b = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a + b - c


def fit_crater_circle(points: np.ndarray, diameter: float, range_noise: float, rng: np.random.Generator) -> tuple[tuple[float, float, float] | None, np.ndarray]:
    negative = points[points[:, 2] < -0.20]
    if len(negative) < 4:
        return None, negative
    eps = max(0.65, diameter * 0.11)
    labels = dbscan(negative[:, :2], eps, 3)
    clusters = [negative[labels == label, :2] for label in sorted(set(labels) - {-1})]
    if not clusters:
        return None, negative
    cluster = max(clusters, key=len)
    if len(cluster) < 4:
        return None, negative
    return ransac_circle(cluster, diameter, range_noise, rng), negative


def detect_and_compute_cr(points: np.ndarray, diameter: float, range_noise: float, rng: np.random.Generator) -> tuple[float, float, int]:
    model, negative = fit_crater_circle(points, diameter, range_noise, rng)
    if model is None:
        return 0.0, 0.0, len(negative)
    cx, cy, radius = model
    true_radius = diameter / 2.0
    dist = math.hypot(cx, cy)
    cr = circle_intersection_area(true_radius, radius, dist) / (math.pi * true_radius * true_radius)
    return 1.0, float(cr), len(negative)


def value_label(value: float) -> str:
    return f"{value:g}m".replace(".", "p")


def plot_crater_completion(
    out_dir: Path,
    *,
    diameter: float,
    ddr: float,
    front_rim_distance: float,
    sensor_height: float,
    horizontal_fov: float,
    vertical_fov: float,
    angular_resolution: float,
    range_noise: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    points = simulate_lidar(
        diameter=diameter,
        ddr=ddr,
        front_rim_distance=front_rim_distance,
        sensor_height=sensor_height,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        angular_resolution=angular_resolution,
        range_noise=range_noise,
        rng=rng,
    )
    model, negative = fit_crater_circle(points, diameter, range_noise, rng)
    radius = diameter / 2.0
    span = max(8.0, diameter * 0.85)
    xs = np.linspace(-span, span, 220)
    ys = np.linspace(-span, span, 220)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = crater_height(xx, yy, diameter, ddr)

    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, shrink=0.82, label="height z (m)")
    if len(negative) > 0:
        ax.scatter(negative[:, 0], negative[:, 1], s=12, c="black", label="negative LiDAR points")
    ax.add_patch(plt.Circle((0.0, 0.0), radius, color="red", fill=False, linewidth=2.2, label="actual crater"))
    if model is not None:
        cx, cy, fit_radius = model
        ax.add_patch(plt.Circle((cx, cy), fit_radius, color="deepskyblue", fill=False, linewidth=2.2, label="completed crater"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Crater completion, D={diameter:g} m, DDR={ddr:.2f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    height_suffix = value_label(sensor_height)
    fig.savefig(out_dir / f"crater_completion_current_sensor_h_{height_suffix}_ddr_{ddr:.2f}.png", dpi=180)
    plt.close(fig)


def distance_label(distance: float) -> str:
    return value_label(distance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/ddr_vs_cr_current")
    parser.add_argument("--diameters", type=float, nargs="+", default=[5.0, 8.0, 10.0])
    parser.add_argument("--front-rim-distance", type=float, default=None)
    parser.add_argument("--front-rim-distances", type=float, nargs="+", default=[15.0, 10.0, 5.0])
    parser.add_argument("--sensor-height", type=float, default=1.5)
    parser.add_argument("--horizontal-fov", type=float, default=54.0)
    parser.add_argument("--vertical-fov", type=float, default=16.0)
    parser.add_argument("--angular-resolution", type=float, default=1.0)
    parser.add_argument("--range-noise", type=float, default=0.02)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--completion-diameter", type=float, default=10.0)
    parser.add_argument("--completion-ddrs", type=float, nargs="+", default=[0.05, 0.25])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ddr_values = np.arange(0.05, 0.501, 0.01)
    front_rim_distances = [args.front_rim_distance] if args.front_rim_distance is not None else args.front_rim_distances
    written_paths = []

    for front_rim_distance in front_rim_distances:
        rows = []
        for diameter in args.diameters:
            for ddr in ddr_values:
                dps = []
                crs = []
                negatives = []
                for trial in range(args.trials):
                    rng = np.random.default_rng(
                        args.seed
                        + trial
                        + int(round(ddr * 1000))
                        + int(round(diameter * 100))
                        + int(round(front_rim_distance * 10))
                    )
                    points = simulate_lidar(
                        diameter=diameter,
                        ddr=float(ddr),
                        front_rim_distance=front_rim_distance,
                        sensor_height=args.sensor_height,
                        horizontal_fov=args.horizontal_fov,
                        vertical_fov=args.vertical_fov,
                        angular_resolution=args.angular_resolution,
                        range_noise=args.range_noise,
                        rng=rng,
                    )
                    dp, cr, negative_count = detect_and_compute_cr(points, diameter, args.range_noise, rng)
                    dps.append(dp)
                    crs.append(cr)
                    negatives.append(negative_count)
                rows.append(
                    {
                        "front_rim_distance": float(front_rim_distance),
                        "sensor_height": float(args.sensor_height),
                        "diameter": float(diameter),
                        "ddr": float(ddr),
                        "dp": float(np.mean(dps)),
                        "cr": float(np.mean(crs)),
                        "negative_points": float(np.mean(negatives)),
                    }
                )

        suffix = distance_label(front_rim_distance)
        height_suffix = value_label(args.sensor_height)
        csv_path = out_dir / f"ddr_vs_cr_current_sensor_h_{height_suffix}_front_rim_{suffix}.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["front_rim_distance", "sensor_height", "diameter", "ddr", "dp", "cr", "negative_points"])
            writer.writeheader()
            writer.writerows(rows)

        fig, ax = plt.subplots(figsize=(6, 4))
        for diameter in args.diameters:
            group = [row for row in rows if row["diameter"] == float(diameter)]
            ax.plot([row["ddr"] for row in group], [row["cr"] for row in group], "o-", label=f"D={diameter:g} m")
        ax.set_xlabel("Depth to diameter ratio (DDR)")
        ax.set_ylabel("Completion rate (CR)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"DDR vs CR, sensor height={args.sensor_height:g} m, front rim distance={front_rim_distance:g} m")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png_path = out_dir / f"ddr_vs_cr_current_sensor_h_{height_suffix}_front_rim_{suffix}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        written_paths.extend([png_path, csv_path])

    for completion_ddr in args.completion_ddrs:
        plot_crater_completion(
            out_dir,
            diameter=args.completion_diameter,
            ddr=completion_ddr,
            front_rim_distance=front_rim_distances[0],
            sensor_height=args.sensor_height,
            horizontal_fov=args.horizontal_fov,
            vertical_fov=args.vertical_fov,
            angular_resolution=args.angular_resolution,
            range_noise=args.range_noise,
            seed=args.seed,
        )

    for path in written_paths:
        print(f"Wrote {path}")
    for completion_ddr in args.completion_ddrs:
        height_suffix = value_label(args.sensor_height)
        print(f"Wrote {out_dir / f'crater_completion_current_sensor_h_{height_suffix}_ddr_{completion_ddr:.2f}.png'}")


if __name__ == "__main__":
    main()
