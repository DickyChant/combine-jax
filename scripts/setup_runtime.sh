#!/usr/bin/env bash
# Build from public sources; do not source ROOT and LCG into one interpreter.
set -eo pipefail
source "${COMBINE_JAX_LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh}"
exec python "$(dirname "$(realpath "$0")")/setup_runtime.py" "$@"
