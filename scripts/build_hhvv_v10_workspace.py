"""Build an exact v4_final mass card with the installed native-v10 physics model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass', type=int, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists()
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    # Serialising the full RooWorkspace traverses a deep graph. The login-node
    # 8 MiB stack can segfault inside ROOT before its crash handler can run.
    old_stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (old_stack[1], old_stack[1]))
    import ROOT as R
    assert R.gROOT.GetVersion() == '6.30/07'
    base = Path(os.environ['CMSSW_BASE'])
    assert base.name == 'CMSSW_14_1_0_pre4'
    repo = Path('/afs/cern.ch/work/s/sqian/crab_inference_dev/inference')
    source = Path('/eos/cms/store/group/phys_b2g/sqian/SHHVV/xinyue/HHVV/HHVV_combCard_singlet_v4_final')/('datacard_mass_'+str(args.mass)+'.txt')
    card = args.directory/'card.txt'
    shutil.copyfile(source, card)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONFAULTHANDLER='1', DHI_COMBINE_VERSION='v10.3.3',
        LAW_HOME=str((args.directory/'law').resolve()), LAW_CONFIG_FILE='/dev/null',
        DHI_HHVV_INPUTS=str(repo/'dhi/models/hhvv_inputs'), DHI_HAS_GFAL='0', DHI_USE_GFAL='0')
    env['PYTHONPATH'] = os.pathsep.join([str(repo), str(repo/'dhi/models'), env.get('PYTHONPATH', '')])
    workspace = args.directory/'workspace.root'
    command = ['text2workspace.py', str(card.resolve()), '-P', 'hhvv_model:model_default',
               '-m', str(args.mass), '-o', str(workspace.resolve())]
    report = dict(status='building', mass=args.mass, source_card=str(source), card_sha256=sha(card),
        root_version=R.gROOT.GetVersion(), combine_version='10.3.3',
        native_library_sha256=sha(base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'),
        script_sha256=sha(__file__), command=command, original_stack_limit=list(old_stack),
        stack_limit=list(resource.getrlimit(resource.RLIMIT_STACK)),
        physics_sources={str(repo/n):sha(repo/n) for n in ['dhi/models/hhvv_model.py', 'dhi/config.py', 'dhi/util.py',
            'dhi/models/hhvv_inputs/XS_ggF.csv', 'dhi/models/hhvv_inputs/BR.csv']},
        scope='Workspace translation only. No fit or observed likelihood evaluation; all outputs stay in project scratch.')
    output = args.directory/'build.json'
    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')
    save(); started = time.perf_counter()
    with (args.directory/'text2workspace.log').open('x') as stream:
        result = subprocess.run(command, cwd=args.directory, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    report.update(returncode=result.returncode, wall_seconds=time.perf_counter()-started,
                  child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
                  log_sha256=sha(args.directory/'text2workspace.log'))
    report['status'] = 'built' if result.returncode == 0 else 'failed'
    if result.returncode == 0:
        report['workspace_sha256'] = sha(workspace)
    save()
    assert result.returncode == 0, 'Native workspace build failed; see recorded log'
    print(report['status'], args.mass, report['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
