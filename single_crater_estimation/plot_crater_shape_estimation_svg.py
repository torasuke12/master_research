from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".python_packages"
SYNTHTERRAIN_SRC = ROOT / "synthterrain-main" / "src" / "python"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
if SYNTHTERRAIN_SRC.exists():
    sys.path.insert(0, str(SYNTHTERRAIN_SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from crater_estimation_core import (
    EstimationParams,
    build_surface_for_degraded_diameter,
    estimate_shape,
    local_surface_height,
    noisy_points_from_hits,
    observed_rim_profile,
    residual_sum_squares,
    simulate_lidar_hits,
)
NPF_COEFFICIENTS = np.array(
    [
        -3.076756,
        -3.557528,
        0.781027,
        1.021521,
        -0.156012,
        -0.444058,
        0.019977,
        0.086850,
        -0.005874,
        -0.006809,
        8.25e-04,
        5.54e-05,
    ]
)


def viper_env_spec_csfd(diameter: np.ndarray) -> np.ndarray:
    """synthterrain.crater.functions.VIPER_Env_Spec.csfd."""
    diameter = np.asarray(diameter, dtype=float)
    out = np.empty_like(diameter, dtype=float)
    out[diameter <= 80] = 29174 * np.float_power(diameter[diameter <= 80], -1.92)
    out[diameter > 80] = 156228 * np.float_power(diameter[diameter > 80], -2.389)
    return out / 1e6


def npf_csfd(diameter: np.ndarray) -> np.ndarray:
    """synthterrain.crater.functions.NPF.csfd for D >= 10 m."""
    diameter = np.asarray(diameter, dtype=float)
    log_d_km = np.log10(diameter / 1000.0)
    poly = np.polynomial.Polynomial(NPF_COEFFICIENTS)
    return np.float_power(10, poly(log_d_km)) / 1e6


def synthterrain_equilibrium_age_years(diameter: float) -> float:
    """synthterrain.crater.age.equilibrium_age using default VIPER/NPF at D=15 m."""
    diameters = np.array([diameter], dtype=float)
    upper_diameters = np.float_power(10, np.log10(diameters) + 0.1)
    eq = viper_env_spec_csfd(diameters) - viper_env_spec_csfd(upper_diameters)
    pf = npf_csfd(diameters) - npf_csfd(upper_diameters)
    return float((1e9 * eq / pf)[0])


def build_params(args: argparse.Namespace) -> EstimationParams:
    max_age_years = synthterrain_equilibrium_age_years(args.diameter)
    if args.true_age_years is not None:
        true_age_years = float(args.true_age_years)
    elif args.true_age_fraction is not None:
        true_age_years = float(args.true_age_fraction * max_age_years)
    else:
        rng = np.random.default_rng(args.true_age_seed)
        true_age_years = float(rng.uniform(0.0, max_age_years))

    return EstimationParams(
        diameter=args.diameter,
        true_age_years=min(true_age_years, max_age_years),
        max_age_years=max_age_years,
        age_candidates=args.age_candidates,
        noise=args.noise,
        background_sigma=args.background_sigma,
        rim_range_min=0.98,
        rim_range_max=1.5,
        normalization_mode="observed",
        lidar_distance=args.lidar_distance,
        lidar_height=args.lidar_height,
        horizontal_fov=args.horizontal_fov,
        vertical_fov=args.vertical_fov,
        angular_resolution=args.horizontal_resolution,
        seed=args.seed,
    )


def write_profile_csv(
    path: Path,
    radius: np.ndarray,
    true_elevation: np.ndarray,
    estimated_elevation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["normalized_radius", "true_elevation_m", "estimated_elevation_m"])
        for row in zip(radius, true_elevation, estimated_elevation):
            writer.writerow([f"{row[0]:.8f}", f"{row[1]:.10f}", f"{row[2]:.10f}"])


def line_profile(surface, normalized_radius: np.ndarray, rim_diameter: float) -> np.ndarray:
    physical_radius = normalized_radius * (rim_diameter / 2)
    return np.array([local_surface_height(surface, float(radius), 0.0) for radius in physical_radius])


def plot_case(args: argparse.Namespace) -> tuple[Path, Path, dict[str, float | int]]:
    params = build_params(args)
    bin_edges = np.linspace(0.0, 1.5, 76)

    true_surface = build_surface_for_degraded_diameter(params.diameter, params.true_age_years)
    lidar_hits = simulate_lidar_hits(true_surface, params)
    points = noisy_points_from_hits(lidar_hits, params)
    observed = observed_rim_profile(points, bin_edges, params)
    observed.degraded_diameter = lidar_hits.degraded_diameter

    candidate_ages = np.linspace(0.0, params.max_age_years, params.age_candidates)
    candidate_surfaces = [
        build_surface_for_degraded_diameter(params.diameter, float(age))
        for age in candidate_ages
    ]
    estimate = estimate_shape(
        params,
        observed,
        bin_edges,
        lidar_hits.degraded_diameter,
        candidate_surfaces=candidate_surfaces,
    )
    if estimate is None:
        raise RuntimeError("No rim points were accepted; estimation could not be performed.")

    profile_radius = np.linspace(0.0, 1.5, 601)
    true_elevation = line_profile(true_surface, profile_radius, lidar_hits.degraded_diameter)
    estimate_elevation = line_profile(estimate.surface, profile_radius, lidar_hits.degraded_diameter)
    residual = residual_sum_squares(
        observed.observed_radius,
        observed.observed_elevation,
        estimate.profile_radius,
        estimate.profile_elevation,
    )
    rmse_m = float(np.sqrt(residual / len(observed.observed_radius)))
    age_error_years = estimate.age_years - params.true_age_years

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.axvspan(0.98, 1.02, color="#f59e0b", alpha=0.22, label="Rim: R=0.98-1.02")
    ax.axvspan(1.02, 1.5, color="#3b82f6", alpha=0.14, label="Exterior: R=1.02-1.5")
    lidar_plot_points = points[(points[:, 3] >= 0.98) & (points[:, 3] <= 1.5)]
    if len(lidar_plot_points):
        ax.scatter(
            lidar_plot_points[:, 3],
            lidar_plot_points[:, 2],
            s=14,
            color="#2563eb",
            alpha=0.55,
            linewidth=0,
            zorder=3,
            label="LiDAR points: R=0.98-1.5",
        )
    ax.plot(profile_radius, true_elevation, color="#111111", lw=2.0, zorder=5, label="True profile")
    ax.plot(
        profile_radius,
        estimate_elevation,
        color="#d43b25",
        lw=2.0,
        ls="--",
        zorder=6,
        label="Estimated profile",
    )
    if len(observed.accepted_points):
        ax.scatter(
            observed.accepted_points[:, 3],
            observed.accepted_points[:, 2],
            s=38,
            facecolor="none",
            edgecolor="#7c2d12",
            alpha=0.95,
            linewidth=1.0,
            zorder=7,
            label="Accepted points for estimation",
        )

    metrics = (
        f"D = {params.diameter:g} m\n"
        f"FOV = {params.horizontal_fov:g} deg x {params.vertical_fov:g} deg\n"
        f"Angular res. = {args.horizontal_resolution:g} deg x {args.vertical_resolution:g} deg\n"
        f"LiDAR height = {params.lidar_height:g} m, distance = {params.lidar_distance:g} m\n"
        f"True age = {params.true_age_years / 1e6:.1f} Myr\n"
        f"Estimated age = {estimate.age_years / 1e6:.1f} Myr\n"
        f"Age error = {age_error_years / 1e6:+.1f} Myr\n"
        f"RMSE = {rmse_m:.4f} m"
    )
    ax.text(
        0.97,
        0.04,
        metrics,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86, "edgecolor": "#d1d5db"},
    )
    ax.set_xlim(0.0, 1.5)
    ax.set_xlabel("Normalized radius R = r / R_rim")
    ax.set_ylabel("Elevation [m]")
    ax.set_title("Crater Shape Estimation Profile")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "crater_shape_profile_D15m_fov100x50_res05_h05_dist15.svg"
    csv_path = args.output_dir / "crater_shape_profile_D15m_fov100x50_res05_h05_dist15.csv"
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    write_profile_csv(csv_path, profile_radius, true_elevation, estimate_elevation)

    summary = {
        "true_age_years": params.true_age_years,
        "max_age_years": params.max_age_years,
        "estimated_age_years": estimate.age_years,
        "age_error_years": age_error_years,
        "rmse_m": rmse_m,
        "accepted_rim_points": int(len(observed.accepted_points)),
        "observed_degraded_diameter_m": float(lidar_hits.degraded_diameter),
    }
    return svg_path, csv_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot crater_shape_estimation_gui.html-equivalent result as SVG."
    )
    parser.add_argument("--diameter", type=float, default=15.0)
    parser.add_argument("--horizontal-fov", type=float, default=100.0)
    parser.add_argument("--vertical-fov", type=float, default=50.0)
    parser.add_argument("--horizontal-resolution", type=float, default=0.5)
    parser.add_argument("--vertical-resolution", type=float, default=0.5)
    parser.add_argument("--lidar-height", type=float, default=0.5)
    parser.add_argument("--lidar-distance", type=float, default=15.0)
    parser.add_argument("--age-candidates", type=int, default=61)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--background-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--true-age-seed",
        type=int,
        default=3,
        help=(
            "Seed for drawing true age from Uniform(0, equilibrium age) when "
            "--true-age-years and --true-age-fraction are omitted."
        ),
    )
    parser.add_argument("--true-age-years", type=float, default=None)
    parser.add_argument(
        "--true-age-fraction",
        type=float,
        default=None,
        help=(
            "True age as a fraction of the synthterrain equilibrium age. "
            "When omitted with --true-age-years, true age is drawn uniformly from [0, equilibrium age]."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "matplotlib_crater_shape_profile",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.vertical_resolution != args.horizontal_resolution:
        raise ValueError(
            "The GUI model has one angularResolution value; use equal horizontal/vertical resolutions."
        )
    svg_path, csv_path, summary = plot_case(args)
    print(f"wrote {svg_path}")
    print(f"wrote {csv_path}")
    print(f"true_age_myr={summary['true_age_years'] / 1e6:.6f}")
    print(f"max_age_myr={summary['max_age_years'] / 1e6:.6f}")
    print(f"estimated_age_myr={summary['estimated_age_years'] / 1e6:.6f}")
    print(f"age_error_myr={summary['age_error_years'] / 1e6:.6f}")
    print(f"rmse_m={summary['rmse_m']:.8f}")
    print(f"accepted_rim_points={summary['accepted_rim_points']}")
    print(f"observed_degraded_diameter_m={summary['observed_degraded_diameter_m']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
