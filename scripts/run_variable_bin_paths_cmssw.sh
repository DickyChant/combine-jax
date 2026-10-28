#!/bin/bash
# Run from /tmp/sqian/combine-jax on an allowed lxplus node. AFS is read-only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
audit_project=/tmp/sqian/combine-jax
audit_reference=/afs/cern.ch/work/s/sqian/crab_inference_dev
mkdir -p "$audit_project/build/variable_bin_path_audit"
for audit_stack in v10 official_v11; do
  (
    set +u
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    if [[ "$audit_stack" == v10 ]]; then
      cd "$audit_reference/software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4/src"
    else
      cd "$audit_reference/software_v11/combine_v11.0.0_el9_amd64_gcc13/CMSSW_16_0_0/src"
    fi
    eval "$(scram runtime -sh)"
    cd "$audit_project"
    python3 scripts/probe_variable_bin_paths.py \
      --plain1d "$audit_reference/v10v11/minimal1d/ws.root" \
      --plain2d "$audit_reference/v10v11/minimal2d/ws.root" \
      --morpher build/variable_bin_repro_v10/asimov.root \
      --output "build/variable_bin_path_audit/$audit_stack.json"
  ) > "$audit_project/build/variable_bin_path_audit/$audit_stack.log" 2>&1
done
