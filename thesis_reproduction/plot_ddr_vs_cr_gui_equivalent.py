#!/usr/bin/env python3
"""Plot DDR vs CR using the same numerical process as interactive_simulator.html."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def angle_samples(fov_deg: float, resolution_deg: float) -> list[float]:
    count = max(1, math.floor(fov_deg / resolution_deg) + 1)
    if count == 1:
        return [0.0]
    span = resolution_deg * (count - 1)
    start = -span / 2.0
    return [math.radians(start + resolution_deg * i) for i in range(count)]


def crater_height(x: float, y: float, diameter: float, depth: float) -> float:
    radius = diameter / 2.0
    rho = math.hypot(x, y)
    if rho <= radius:
        return (rho * rho) * depth / (radius**2) - depth
    return 0.0


def rand_noise(index: int, sigma: float) -> float:
    x = math.sin(index * 927.13 + 37.7) * 43758.5453
    y = math.sin(index * 311.91 + 11.3) * 24634.6345
    u1 = max(1e-6, x - math.floor(x))
    u2 = y - math.floor(y)
    return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def ray_ground_hit(
    diameter: float,
    depth: float,
    sensor: tuple[float, float, float],
    h: float,
    v: float,
    max_range: float,
) -> float | None:
    sx, sy, sz = sensor
    step = 0.12
    previous = 0.5
    r = previous
    while r <= max_range:
        x = sx + r * math.cos(v) * math.sin(h)
        y = sy + r * math.cos(v) * math.cos(h)
        z = sz + r * math.sin(v)
        if z <= crater_height(x, y, diameter, depth):
            lo = previous
            hi = r
            for _ in range(8):
                mid = (lo + hi) / 2.0
                mx = sx + mid * math.cos(v) * math.sin(h)
                my = sy + mid * math.cos(v) * math.cos(h)
                mz = sz + mid * math.sin(v)
                if mz <= crater_height(mx, my, diameter, depth):
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
) -> list[dict[str, float | int]]:
    radius = diameter / 2.0
    depth = ddr * diameter
    sensor = (0.0, -(front_rim_distance + radius), sensor_height)
    h_samples = angle_samples(horizontal_fov, angular_resolution)
    v_samples = angle_samples(vertical_fov, angular_resolution)
    pitch_center = math.atan2(-sensor_height, front_rim_distance + radius)
    max_range = min(75.0, front_rim_distance + diameter * 2.0 + 8.0)
    points: list[dict[str, float | int]] = []
    ray_index = 0
    sx, sy, sz = sensor

    for vi, v_offset in enumerate(v_samples):
        v = pitch_center + v_offset
        for hi, h in enumerate(h_samples):
            hit_range = ray_ground_hit(diameter, depth, sensor, h, v, max_range)
            if hit_range is not None:
                measured = max(0.0, hit_range + rand_noise(ray_index, range_noise))
                points.append(
                    {
                        "x": sx + measured * math.cos(v) * math.sin(h),
                        "y": sy + measured * math.cos(v) * math.cos(h),
                        "z": sz + measured * math.sin(v),
                        "h": h,
                        "v": v,
                        "hi": hi,
                        "vi": vi,
                        "ray_index": ray_index,
                    }
                )
            ray_index += 1
    return points


def dbscan(points: list[dict[str, float | int]], eps: float, min_samples: int) -> list[int]:
    labels = [-1] * len(points)
    visited = [False] * len(points)
    cluster_id = 0

    def neighbors_of(index: int) -> list[int]:
        p0 = points[index]
        return [
            i
            for i, point in enumerate(points)
            if math.hypot(float(point["x"]) - float(p0["x"]), float(point["y"]) - float(p0["y"])) <= eps
        ]

    for index in range(len(points)):
        if visited[index]:
            continue
        visited[index] = True
        neighbors = neighbors_of(index)
        if len(neighbors) < min_samples:
            continue

        labels[index] = cluster_id
        seeds = list(neighbors)
        while seeds:
            current = seeds.pop()
            if not visited[current]:
                visited[current] = True
                current_neighbors = neighbors_of(current)
                if len(current_neighbors) >= min_samples:
                    seeds.extend(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1
    return labels


def largest_cluster(points: list[dict[str, float | int]], labels: list[int]) -> list[dict[str, float | int]]:
    clusters: dict[int, list[dict[str, float | int]]] = {}
    for point, label in zip(points, labels, strict=True):
        if label < 0:
            continue
        clusters.setdefault(label, []).append(point)
    best: list[dict[str, float | int]] = []
    for cluster in clusters.values():
        if len(cluster) > len(best):
            best = cluster
    return best


def circle_from_3(
    a: dict[str, float | int],
    b: dict[str, float | int],
    c: dict[str, float | int],
) -> tuple[float, float, float] | None:
    x1, y1 = float(a["x"]), float(a["y"])
    x2, y2 = float(b["x"]), float(b["y"])
    x3, y3 = float(c["x"]), float(c["y"])
    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 1e-8:
        return None
    aa = x1 * x1 + y1 * y1
    bb = x2 * x2 + y2 * y2
    cc = x3 * x3 + y3 * y3
    cx = (aa * (y2 - y3) + bb * (y3 - y1) + cc * (y1 - y2)) / det
    cy = (aa * (x3 - x2) + bb * (x1 - x3) + cc * (x2 - x1)) / det
    radius = math.hypot(x1 - cx, y1 - cy)
    if not math.isfinite(radius):
        return None
    return cx, cy, radius


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    a = [row[:] for row in matrix]
    b = vector[:]
    for col in range(3):
        pivot = col
        for row in range(col + 1, 3):
            if abs(a[row][col]) > abs(a[pivot][col]):
                pivot = row
        if abs(a[pivot][col]) < 1e-10:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        div = a[col][col]
        for j in range(col, 3):
            a[col][j] /= div
        b[col] /= div
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 3):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]
    return b


def least_squares_circle(points: list[dict[str, float | int]]) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    sxx = syy = sxy = sx = sy = 0.0
    bx = by = b0 = 0.0
    for point in points:
        x = float(point["x"])
        y = float(point["y"])
        row0 = 2.0 * x
        row1 = 2.0 * y
        rhs = x * x + y * y
        sxx += row0 * row0
        syy += row1 * row1
        sxy += row0 * row1
        sx += row0
        sy += row1
        bx += row0 * rhs
        by += row1 * rhs
        b0 += rhs
    solution = solve_3x3(
        [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(len(points))]],
        [bx, by, b0],
    )
    if solution is None:
        return None
    cx, cy, c = solution
    radius = math.sqrt(max(0.0, c + cx * cx + cy * cy))
    if not math.isfinite(radius):
        return None
    return cx, cy, radius


def angular_span(points: list[dict[str, float | int]], circle: tuple[float, float, float]) -> float:
    cx, cy, _ = circle
    angles = sorted(math.atan2(float(point["y"]) - cy, float(point["x"]) - cx) for point in points)
    if len(angles) < 2:
        return 0.0
    max_gap = 0.0
    for index in range(1, len(angles)):
        max_gap = max(max_gap, angles[index] - angles[index - 1])
    max_gap = max(max_gap, angles[0] + math.pi * 2.0 - angles[-1])
    return math.pi * 2.0 - max_gap


def ransac_circle(
    points: list[dict[str, float | int]],
    diameter: float,
    range_noise: float,
) -> tuple[float, float, float] | None:
    if len(points) < 4:
        return None
    residual_threshold = max(0.18, diameter * 0.035 + range_noise * 2.0)
    min_radius = max(0.7, diameter * 0.18)
    max_radius = max(3.0, diameter * 1.25)
    best: tuple[float, float, float] | None = None
    best_inliers: list[dict[str, float | int]] = []
    best_score = -math.inf
    iterations = min(900, max(180, len(points) * len(points) * 2))

    for index in range(iterations):
        i1 = math.floor((((math.sin(index * 12.9898) * 43758.5453) % 1.0 + 1.0) % 1.0) * len(points))
        i2 = math.floor((((math.sin((index + 17) * 78.233) * 23171.629) % 1.0 + 1.0) % 1.0) * len(points))
        i3 = math.floor((((math.sin((index + 43) * 37.719) * 91573.137) % 1.0 + 1.0) % 1.0) * len(points))
        if i1 == i2 or i1 == i3 or i2 == i3:
            continue
        model = circle_from_3(points[i1], points[i2], points[i3])
        if model is None or model[2] < min_radius or model[2] > max_radius:
            continue

        cx, cy, radius = model
        inliers: list[dict[str, float | int]] = []
        residual_sum = 0.0
        for point in points:
            residual = abs(math.hypot(float(point["x"]) - cx, float(point["y"]) - cy) - radius)
            if residual <= residual_threshold:
                inliers.append(point)
                residual_sum += residual
        if len(inliers) < 4:
            continue
        mean_residual = residual_sum / len(inliers)
        span = angular_span(inliers, model)
        score = len(inliers) * 20.0 + span * 8.0 - mean_residual * 100.0
        if score > best_score:
            best_score = score
            best = model
            best_inliers = inliers

    if best is None or len(best_inliers) < 4:
        return None
    refined = least_squares_circle(best_inliers) or best
    if refined[2] < min_radius or refined[2] > max_radius:
        return best
    return refined


def fit_circle(
    points: list[dict[str, float | int]],
    diameter: float,
    range_noise: float,
    negative_threshold: float,
) -> tuple[tuple[float, float, float] | None, list[dict[str, float | int]]]:
    negative = [point for point in points if float(point["z"]) < negative_threshold]
    if len(negative) < 4:
        return None, negative
    eps = max(0.65, diameter * 0.11)
    labels = dbscan(negative, eps, 3)
    cluster = largest_cluster(negative, labels)
    if len(cluster) < 4:
        return None, negative
    return ransac_circle(cluster, diameter, range_noise), negative


def circle_intersection_area(r1: float, r2: float, distance: float) -> float:
    if not math.isfinite(r2) or r2 <= 0.0:
        return 0.0
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = r1 * r1 * math.acos((distance * distance + r1 * r1 - r2 * r2) / (2.0 * distance * r1))
    b = r2 * r2 * math.acos((distance * distance + r2 * r2 - r1 * r1) / (2.0 * distance * r2))
    c = 0.5 * math.sqrt(
        (-distance + r1 + r2)
        * (distance + r1 - r2)
        * (distance - r1 + r2)
        * (distance + r1 + r2)
    )
    return a + b - c


def detect_and_compute_cr(
    *,
    diameter: float,
    ddr: float,
    front_rim_distance: float,
    sensor_height: float,
    horizontal_fov: float,
    vertical_fov: float,
    angular_resolution: float,
    range_noise: float,
    negative_threshold: float,
) -> tuple[float, float, int, float]:
    points = simulate_lidar(
        diameter=diameter,
        ddr=ddr,
        front_rim_distance=front_rim_distance,
        sensor_height=sensor_height,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        angular_resolution=angular_resolution,
        range_noise=range_noise,
    )
    model, negative = fit_circle(points, diameter, range_noise, negative_threshold)
    if model is None:
        return 0.0, 0.0, len(negative), 0.0
    cx, cy, radius = model
    true_radius = diameter / 2.0
    distance = math.hypot(cx, cy)
    cr = circle_intersection_area(true_radius, radius, distance) / (math.pi * true_radius * true_radius)
    return 1.0, max(0.0, min(1.0, cr)), len(negative), radius * 2.0


def value_label(value: float) -> str:
    return f"{value:g}m".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/ddr_vs_cr_gui_equivalent")
    parser.add_argument("--diameters", type=float, nargs="+", default=[5.0, 8.0, 10.0])
    parser.add_argument("--front-rim-distance", type=float, default=15.0)
    parser.add_argument("--sensor-height", type=float, default=1.5)
    parser.add_argument("--horizontal-fov", type=float, default=54.0)
    parser.add_argument("--vertical-fov", type=float, default=16.0)
    parser.add_argument("--angular-resolution", type=float, default=1.0)
    parser.add_argument("--range-noise", type=float, default=0.02)
    parser.add_argument("--negative-threshold", type=float, default=-0.10)
    parser.add_argument("--ddr-min", type=float, default=0.05)
    parser.add_argument("--ddr-max", type=float, default=0.50)
    parser.add_argument("--ddr-step", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ddr_values = []
    index = 0
    while True:
        ddr = args.ddr_min + args.ddr_step * index
        if ddr > args.ddr_max + args.ddr_step * 0.5:
            break
        ddr_values.append(round(ddr, 10))
        index += 1

    rows = []
    for diameter in args.diameters:
        for ddr in ddr_values:
            dp, cr, negative_points, fit_diameter = detect_and_compute_cr(
                diameter=float(diameter),
                ddr=float(ddr),
                front_rim_distance=float(args.front_rim_distance),
                sensor_height=float(args.sensor_height),
                horizontal_fov=float(args.horizontal_fov),
                vertical_fov=float(args.vertical_fov),
                angular_resolution=float(args.angular_resolution),
                range_noise=float(args.range_noise),
                negative_threshold=float(args.negative_threshold),
            )
            rows.append(
                {
                    "front_rim_distance": float(args.front_rim_distance),
                    "sensor_height": float(args.sensor_height),
                    "diameter": float(diameter),
                    "ddr": float(ddr),
                    "dp": dp,
                    "cr": cr,
                    "negative_points": negative_points,
                    "fit_diameter": fit_diameter,
                }
            )

    suffix = value_label(args.front_rim_distance)
    height_suffix = value_label(args.sensor_height)
    csv_path = out_dir / f"ddr_vs_cr_gui_equivalent_sensor_h_{height_suffix}_front_rim_{suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "front_rim_distance",
                "sensor_height",
                "diameter",
                "ddr",
                "dp",
                "cr",
                "negative_points",
                "fit_diameter",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for diameter in args.diameters:
        group = [row for row in rows if row["diameter"] == float(diameter)]
        ax.plot([row["ddr"] for row in group], [row["cr"] for row in group], "o-", label=f"D={diameter:g} m")
    ax.set_xlabel("Depth to diameter ratio (DDR)")
    ax.set_ylabel("Completion rate (CR)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"GUI-equivalent DDR vs CR, sensor height={args.sensor_height:g} m, "
        f"front rim distance={args.front_rim_distance:g} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"ddr_vs_cr_gui_equivalent_sensor_h_{height_suffix}_front_rim_{suffix}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
