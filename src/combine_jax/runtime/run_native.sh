#!/usr/bin/env bash
set -eo pipefail
native_stack="$1"; shift
native_work="$PWD"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH VIRTUAL_ENV
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$native_stack/src"
eval "$(scram runtime -sh)"
cd "$native_work"
exec python3 "$@"
