#!/usr/bin/env bash
# Generation-only v10 control, isolated from the v11 Python/ROOT process.
set -eo pipefail
test "$(hostname -s)" != lxplus905
test "$PWD" = /tmp/sqian/combine-jax/build/vv4q_v10_asimov_bonly
test ! -e generation.time
native_stack=/afs/cern.ch/work/s/sqian/crab_inference_dev/software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4
source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$native_stack/src" >/dev/null
eval "$(scram runtime -sh)"
popd >/dev/null
export PATH="$native_stack/bin/el9_amd64_gcc12:$PATH"
export LD_LIBRARY_PATH="$native_stack/lib/el9_amd64_gcc12:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - "$native_stack" <<'PY'
import hashlib,json,os,pathlib,platform,shutil,subprocess,sys
stack=pathlib.Path(sys.argv[1]);lib=stack/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
executable=stack/'bin/el9_amd64_gcc12/combine'
assert pathlib.Path(shutil.which('combine'))==executable
resolved=subprocess.check_output(['ldd',str(executable)],text=True)
assert str(lib) in resolved
assert '/lcg/' not in resolved and '6.40' not in resolved
def sha(p):
    with open(p,'rb') as f:return hashlib.sha256(f.read()).hexdigest()
report=dict(status='generation_only',host=platform.node(),cmssw=str(stack),
    root_version=subprocess.check_output(['root-config','--version'],text=True).strip(),
    executable=str(executable),executable_sha256=sha(executable),resolved_libraries=resolved,
    library_sha256=sha(lib),workspace_sha256=sha('workspace.root'),
    asimov_source_sha256=sha(stack/'src/HiggsAnalysis/CombinedLimit/src/AsimovUtils.cc'),
    toymcsampler_source_sha256=sha(stack/'src/HiggsAnalysis/CombinedLimit/src/ToyMCSamplerOpt.cc'),
    scope='Background-only pre-fit Asimov generation; no observed-data fit, no limit fit, no v11 ROOT library loaded.')
pathlib.Path('environment.json').write_text(json.dumps(report,indent=2)+'\n')
PY
/usr/bin/time -p -o generation.time taskset -c "$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')" \
  "$native_stack/bin/el9_amd64_gcc12/combine" --method GenerateOnly workspace.root --mass 125 --seed 0 --toys -1 \
  --saveToys --redefineSignalPOIs r_hhvv \
  --setParameterRanges MX=1000,6000 --setParameters MX=4000,r_hhvv=0 \
  --X-rtd REMOVE_CONSTANT_ZERO_POINT=1 --X-rtd TMCSO_AdaptivePseudoAsimov=0 \
  --X-rtd TMCSO_PseudoAsimov=0 --name V10Bonly > combine.log 2>&1
