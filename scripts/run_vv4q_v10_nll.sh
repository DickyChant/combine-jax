#!/usr/bin/env bash
set -eo pipefail
test "$(hostname -s)" != lxplus905
test "$PWD" = /tmp/sqian/combine-jax/build/vv4q_v10_asimov_bonly
native_stack=/afs/cern.ch/work/s/sqian/crab_inference_dev/software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$native_stack/src" >/dev/null
eval "$(scram runtime -sh)"
popd >/dev/null
export PATH="$native_stack/bin/el9_amd64_gcc12:$PATH"
export LD_LIBRARY_PATH="$native_stack/lib/el9_amd64_gcc12:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')" \
  python3 export_vv4q_version_nll.py --stack v10 --workspace workspace.root \
  --data higgsCombineV10Bonly.GenerateOnly.mH125.0.root --points vv4q_version_points.json \
  --output shared_nll.json > shared_nll.log 2>&1
