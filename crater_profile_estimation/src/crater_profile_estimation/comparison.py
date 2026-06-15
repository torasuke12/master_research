"""Compare a synthterrain crater with the profile estimator."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "crater-profile-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .estimator import estimate_profile
from .lookup import DEFAULT_LOOKUP_PATH
from .profile import stopar_fresh_dd

DEFAULT_PLOT = Path("output/synthterrain_estimation_comparison.png")
DEFAULT_PROFILE_PLOT = Path(
    "output/synthterrain_estimation_profile_equal_scale.png"
)
DEFAULT_RIGHT_PLOT = Path("output/synthterrain_estimation_profile.png")
DEFAULT_CSV = Path("output/synthterrain_estimation_comparison.csv")
DEFAULT_SUMMARY = Path("output/synthterrain_estimation_comparison.json")


def _load_synthterrain(source: Path):
    package_root = source / "src" / "python"
    if not package_root.is_dir():
        raise FileNotFoundError(f"synthterrain source not found: {package_root}")
    sys.path.insert(0, str(package_root))
    return importlib.import_module("synthterrain.crater.diffusion")


def measure_inner_wall_slope(
    x_m: np.ndarray,
    height_m: np.ndarray,
    diameter_m: float,
    inner_fraction: float = 0.2,
    outer_fraction: float = 0.9,
) -> tuple[float, np.ndarray]:
    """Fit both radial walls over 0.2R--0.9R and return mean angle."""
    radius = diameter_m / 2.0
    radial = np.abs(x_m)
    mask = (radial >= inner_fraction * radius) & (radial <= outer_fraction * radius)
    angles = []
    for side in (x_m < 0, x_m > 0):
        selected = mask & side
        slope = np.polyfit(radial[selected], height_m[selected], 1)[0]
        angles.append(math.degrees(math.atan(abs(slope))))
    return float(np.mean(angles)), mask


def compare(
    diameter_m: float,
    lambda_true: float,
    sigma_slope_deg: float,
    lookup: Path,
    synthterrain_source: Path,
    domain_size: int,
    output_plot: Path,
    output_profile_plot: Path | None,
    output_csv: Path,
    output_summary: Path,
    output_right_plot: Path | None = None,
) -> dict[str, object]:
    if not 1.0 <= diameter_m < 40.0:
        raise ValueError("comparison diameter must satisfy 1 <= D < 40 m")
    diffusion = _load_synthterrain(synthterrain_source)
    true_s_m2 = lambda_true * diameter_m * diameter_m
    age_years = true_s_m2 / diffusion.kappa_diffusivity(diameter_m)
    initial_dd = stopar_fresh_dd(diameter_m)
    true_dd, surface = diffusion.diffuse_d_over_D(
        diameter_m,
        age_years,
        domain_size=domain_size,
        start_dd_adjust=initial_dd,
        return_surface=True,
    )

    axis_fraction = np.linspace(-2.0, 2.0, domain_size)
    x_m = axis_fraction * diameter_m / 2.0
    center = domain_size // 2
    true_height = surface[center, :]
    slope_deg, slope_mask = measure_inner_wall_slope(
        x_m, true_height, diameter_m
    )
    estimate = estimate_profile(
        diameter_m,
        slope_deg=slope_deg,
        sigma_slope_deg=sigma_slope_deg,
        lookup=lookup,
    )

    radius = estimate.radius_m
    estimate_x = np.concatenate((-radius[:0:-1], radius))

    def symmetric(values):
        return np.concatenate((values[:0:-1], values))

    q10 = symmetric(estimate.height_q10_m)
    q50 = symmetric(estimate.height_q50_m)
    q90 = symmetric(estimate.height_q90_m)
    comparison_mask = np.abs(x_m) <= 1.5 * diameter_m / 2.0
    true_x = x_m[comparison_mask]
    true_profile = true_height[comparison_mask]
    estimated_on_true = np.interp(true_x, estimate_x, q50)
    rmse_m = float(np.sqrt(np.mean((estimated_on_true - true_profile) ** 2)))

    fig, (map_ax, ax) = plt.subplots(
        1, 2, figsize=(14, 6.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )
    terrain = map_ax.imshow(
        surface,
        origin="lower",
        extent=(-diameter_m, diameter_m, -diameter_m, diameter_m),
        cmap="terrain",
        aspect="equal",
    )
    map_ax.axhline(0.0, color="white", linewidth=1.0, alpha=0.8)
    map_ax.set_title("synthterrain elevation field")
    map_ax.set_xlabel("x (m)")
    map_ax.set_ylabel("y (m)")
    fig.colorbar(terrain, ax=map_ax, label="Elevation (m)", shrink=0.82)

    ax.fill_between(
        estimate_x, q10, q90, color="#4C78A8", alpha=0.24, label="Estimated 10-90%"
    )
    ax.plot(
        estimate_x,
        q50,
        color="#1F4E79",
        linewidth=2.2,
        label="Estimated median",
    )
    ax.plot(
        true_x,
        true_profile,
        color="#E45756",
        linewidth=2.0,
        label="synthterrain profile",
    )
    slope_plot = slope_mask & comparison_mask
    ax.scatter(
        x_m[slope_plot][:: max(1, slope_plot.sum() // 30)],
        true_height[slope_plot][:: max(1, slope_plot.sum() // 30)],
        s=10,
        color="#E45756",
        alpha=0.45,
        label="Slope-fit samples",
    )
    ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle="--")
    ax.axvline(-diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":")
    ax.axvline(diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":")
    ax.set_title(
        "Central Profile Comparison\n"
        f"D = {diameter_m:g} m, measured S = {slope_deg:.2f} deg, "
        f"sigma = {sigma_slope_deg:g} deg"
    )
    ax.set_xlabel("Distance from crater center (m)")
    ax.set_ylabel("Elevation relative to surrounding surface (m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.03,
        (
            f"true d/D = {true_dd:.3f}, estimated d/D = {estimate.current_dd_median:.3f}\n"
            f"true kappa*t = {true_s_m2:.3g} m^2, estimated = {estimate.diffusion_amount_m2:.3g} m^2\n"
            f"profile RMSE = {rmse_m:.3f} m"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.suptitle("synthterrain Crater and crater_profile_estimation Result")
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=180)
    plt.close(fig)

    if output_right_plot is not None:
        right_fig, right_ax = plt.subplots(
            figsize=(8.4, 6.2), constrained_layout=True
        )
        right_ax.fill_between(
            estimate_x,
            q10,
            q90,
            color="#4C78A8",
            alpha=0.24,
            label="Estimated 10-90%",
        )
        right_ax.plot(
            estimate_x,
            q50,
            color="#1F4E79",
            linewidth=2.2,
            label="Estimated median",
        )
        right_ax.plot(
            true_x,
            true_profile,
            color="#E45756",
            linewidth=2.0,
            label="synthterrain profile",
        )
        right_ax.scatter(
            x_m[slope_plot][:: max(1, slope_plot.sum() // 30)],
            true_height[slope_plot][:: max(1, slope_plot.sum() // 30)],
            s=10,
            color="#E45756",
            alpha=0.45,
            label="Slope-fit samples",
        )
        right_ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle="--")
        right_ax.axvline(
            -diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":"
        )
        right_ax.axvline(
            diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":"
        )
        right_ax.set_title(
            "Central Profile Comparison\n"
            f"D = {diameter_m:g} m, measured S = {slope_deg:.2f} deg, "
            f"sigma = {sigma_slope_deg:g} deg"
        )
        right_ax.set_xlabel("Distance from crater center (m)")
        right_ax.set_ylabel("Elevation relative to surrounding surface (m)")
        right_ax.grid(alpha=0.25)
        right_ax.legend(loc="best")
        right_ax.text(
            0.02,
            0.03,
            (
                f"true d/D = {true_dd:.3f}, estimated d/D = "
                f"{estimate.current_dd_median:.3f}\n"
                f"true kappa*t = {true_s_m2:.3g} m^2, estimated = "
                f"{estimate.diffusion_amount_m2:.3g} m^2\n"
                f"profile RMSE = {rmse_m:.3f} m"
            ),
            transform=right_ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        output_right_plot.parent.mkdir(parents=True, exist_ok=True)
        right_fig.savefig(output_right_plot, dpi=180)
        plt.close(right_fig)

    if output_profile_plot is not None:
        profile_fig, (profile_ax, info_ax) = plt.subplots(
            1,
            2,
            figsize=(11, 5.8),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [4.6, 1.4]},
        )
        profile_ax.fill_between(
            estimate_x,
            q10,
            q90,
            color="#4C78A8",
            alpha=0.24,
            label="Estimated 10-90%",
        )
        profile_ax.plot(
            estimate_x,
            q50,
            color="#1F4E79",
            linewidth=2.2,
            label="Estimated median",
        )
        profile_ax.plot(
            true_x,
            true_profile,
            color="#E45756",
            linewidth=2.0,
            label="synthterrain profile",
        )
        profile_ax.scatter(
            x_m[slope_plot][:: max(1, slope_plot.sum() // 30)],
            true_height[slope_plot][:: max(1, slope_plot.sum() // 30)],
            s=10,
            color="#E45756",
            alpha=0.45,
            label="Slope-fit samples",
        )
        profile_ax.axhline(0.0, color="0.45", linewidth=1.0, linestyle="--")
        profile_ax.axvline(
            -diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":"
        )
        profile_ax.axvline(
            diameter_m / 2.0, color="0.65", linewidth=0.8, linestyle=":"
        )
        profile_ax.set_title(
            "Central Profile Comparison (Equal Scale)\n"
            f"D = {diameter_m:g} m, measured S = {slope_deg:.2f} deg, "
            f"sigma = {sigma_slope_deg:g} deg"
        )
        profile_ax.set_xlabel("Distance from crater center (m)")
        profile_ax.set_ylabel("Elevation relative to surrounding surface (m)")
        profile_ax.set_aspect("equal", adjustable="box")
        profile_ax.grid(alpha=0.25)
        handles, labels = profile_ax.get_legend_handles_labels()
        info_ax.axis("off")
        info_ax.legend(handles, labels, loc="upper left", borderaxespad=0.0)
        info_ax.text(
            0.0,
            0.48,
            (
                f"true d/D = {true_dd:.3f}, estimated d/D = "
                f"{estimate.current_dd_median:.3f}\n"
                f"true kappa*t = {true_s_m2:.3g} m^2, estimated = "
                f"{estimate.diffusion_amount_m2:.3g} m^2\n"
                f"profile RMSE = {rmse_m:.3f} m"
            ),
            transform=info_ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        output_profile_plot.parent.mkdir(parents=True, exist_ok=True)
        profile_fig.savefig(output_profile_plot, dpi=180)
        plt.close(profile_fig)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "distance_m",
                "synthterrain_height_m",
                "estimated_q10_m",
                "estimated_q50_m",
                "estimated_q90_m",
            ]
        )
        writer.writerows(
            zip(
                true_x,
                true_profile,
                np.interp(true_x, estimate_x, q10),
                estimated_on_true,
                np.interp(true_x, estimate_x, q90),
            )
        )

    summary = {
        "diameter_input_m": diameter_m,
        "measured_slope_deg": slope_deg,
        "sigma_slope_deg": sigma_slope_deg,
        "slope_fit_radial_fraction": [0.2, 0.9],
        "synthterrain_initial_dd": initial_dd,
        "synthterrain_current_dd": float(true_dd),
        "synthterrain_diffusion_amount_m2": true_s_m2,
        "synthterrain_age_years": age_years,
        "estimated_current_dd": estimate.current_dd_median,
        "estimated_diffusion_amount_m2": estimate.diffusion_amount_m2,
        "profile_rmse_m": rmse_m,
        "quality_flags": estimate.quality_flags,
        "plot": str(output_plot),
        "equal_scale_profile_plot": (
            str(output_profile_plot) if output_profile_plot is not None else None
        ),
        "right_profile_plot": (
            str(output_right_plot) if output_right_plot is not None else None
        ),
        "csv": str(output_csv),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--diameter", type=float, default=20.0)
    result.add_argument("--lambda-true", type=float, default=0.015)
    result.add_argument("--sigma-slope", type=float, default=2.0)
    result.add_argument("--domain-size", type=int, default=201)
    result.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    result.add_argument("--synthterrain-source", type=Path, default=Path("synthterrain-main"))
    result.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    result.add_argument(
        "--profile-plot", type=Path, default=DEFAULT_PROFILE_PLOT
    )
    result.add_argument("--right-plot", type=Path, default=DEFAULT_RIGHT_PLOT)
    result.add_argument("--output", type=Path, default=DEFAULT_CSV)
    result.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = compare(
        diameter_m=args.diameter,
        lambda_true=args.lambda_true,
        sigma_slope_deg=args.sigma_slope,
        lookup=args.lookup,
        synthterrain_source=args.synthterrain_source,
        domain_size=args.domain_size,
        output_plot=args.plot,
        output_profile_plot=args.profile_plot,
        output_csv=args.output,
        output_summary=args.summary,
        output_right_plot=args.right_plot,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
