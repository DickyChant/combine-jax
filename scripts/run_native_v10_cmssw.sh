#!/bin/bash
# Separate process: never source CUDA-view ROOT into the native reference.
set -eo pipefail
native_work="$PWD"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source /cvmfs/cms.cern.ch/cmsset_default.sh
native_stack="${COMBINE_JAX_NATIVE_CMSSW:-${CMSSW_BASE:-}}"
if [ -z "$native_stack" ]; then
    echo 'Set COMBINE_JAX_NATIVE_CMSSW to a Combine v10.3.3 CMSSW release.' >&2
    exit 2
fi
cd "$native_stack/src"
eval "$(scram runtime -sh)"
cd "$native_work"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$@"
