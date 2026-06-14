#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

export PYTHONPATH="$script_dir/.python_packages${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

jobs="${JOBS:-2}"
for sensor_height in 0.5 1.0 1.5; do
  for front_rim_distance in 5 10 15; do
    python3 plot_ddr_vs_cr_current.py \
      --sensor-height "$sensor_height" \
      --front-rim-distance "$front_rim_distance" \
      --jobs "$jobs"
  done
done
