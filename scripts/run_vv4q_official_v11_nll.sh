#!/usr/bin/env bash
set -eo pipefail
test "$(hostname -s)" != lxplus905
test "$PWD" = /tmp/sqian/combine-jax/build/vv4q_official_v11_probe_20261023
native_stack=/afs/cern.ch/work/s/sqian/crab_inference_dev/software_v11/combine_v11.0.0_el9_amd64_gcc13/CMSSW_16_0_0
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$native_stack/src" >/dev/null
eval "$(scram runtime -sh)"
popd >/dev/null
export PATH="$native_stack/bin/el9_amd64_gcc13:$PATH"
export LD_LIBRARY_PATH="$native_stack/lib/el9_amd64_gcc13:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')" \
  python3 export_vv4q_version_nll.py --stack v11_official --workspace workspace.root \
  --data higgsCombineV10Bonly.GenerateOnly.mH125.0.root --points vv4q_version_points.json \
  --output shared_nll.json > shared_nll.log 2>&1
