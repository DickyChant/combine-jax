#!/usr/bin/env bash
# Example HTCondor worker: code/runtime and neutral model are transferred archives.
set -euo pipefail
runtime_archive="$1"
model_archive="$2"
scan_poi="$3"
shift 3
mkdir runtime results
tar -xzf "$runtime_archive" -C runtime
tar -xzf "$model_archive"  # contains model/model.json and model/arrays.npz
source runtime/setup.sh
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv > results/gpu.txt
# The launcher selects the CUDA view; no ROOT is imported in this process.
combine-jax scan model --device gpu --poi "$scan_poi" --points "$@" \
  --cache compilation-cache -o results/scan.json
