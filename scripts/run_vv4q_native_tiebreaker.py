"""Run v11 on the JAX source workspace with the exact blinded dhi options."""
import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

import uproot
from tensor_benchmark_common import save_json, sha


def process_tree_rss(pid):
    """Linux RSS sampling without adding a numerical-runtime dependency."""
    try:
        status = Path(f'/proc/{pid}/status').read_text()
        own = next(int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:'))
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        return own + sum(process_tree_rss(int(p)) for p in children)
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return 0


def main():
    directory = Path('build/asymptotic_vv4q_noVH_MX4000_native_v11')
    output = Path('reports/asymptotic_vv4q_noVH_MX4000_native_v11.json')
    assert not directory.exists() and not output.exists()
    directory.mkdir()
    recipe_path = Path('reports/asymptotic_vv4q_noVH_MX4000_dhi_command.json')
    recipe = json.loads(recipe_path.read_text())
    command = recipe['command'].copy()
    original_workspace = command[3]
    assert original_workspace == recipe['workspace']
    workspace = Path('build/asymptotic_vv4q_inputs/noVH_4000/workspace.root')
    command[3] = str(workspace.resolve())
    assert command[command.index('--run')+1] == 'expected'
    assert '--noFitAsimov' in command and command[command.index('--toys')+1] == '-1'
    assert version('higgs-combine-tool') == '11.0.0'
    core = min(os.sched_getaffinity(0))
    command = ['taskset', '-c', str(core), *command]
    report = dict(status='waiting_for_serial_jax_completion', script_sha256=sha(__file__),
        command=command, recipe_sha256=sha(recipe_path), workspace=str(workspace),
        workspace_sha256=sha(workspace), combine_version=version('higgs-combine-tool'),
        root_package_version=version('root'), host=os.uname().nodename, cpu_affinity=[core],
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS')},
        scope='Blinded expected v11 tie-breaker on byte-identical JAX source workspace. Exact dhi options, including strategy 0 and fallback tolerances, are preserved. No observed fit.',
        changes_from_dhi=['Combine v11/ROOT 6.40 instead of v10.3.3/CMSSW_14_1_0_pre4',
            'JAX source workspace built with v11 instead of dhi source workspace; separate convention audit records differences.',
            'Local single CPU core instead of the lxbatch reference host.'])
    save_json(output,report)
    jax_report = Path('reports/asymptotic_vv4q_noVH_MX4000.json')
    while json.loads(jax_report.read_text())['status'] in ('preparing','fitting'):
        time.sleep(5)
    report.update(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(output,report)
    start = time.perf_counter()
    with (directory/'combine.log').open('x') as log:
        process = subprocess.Popen(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT)
        report['pid'] = process.pid
        peak_rss = 0
        while process.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            report.update(elapsed_seconds=time.perf_counter()-start,peak_rss_mib=peak_rss/2**20)
            save_json(output,report)
            time.sleep(5)
    report.update(returncode=process.returncode,wall_seconds=time.perf_counter()-start,
        status='failed_execution' if process.returncode else 'completed',log=str(directory/'combine.log'),
        log_sha256=sha(directory/'combine.log'))
    if process.returncode == 0:
        rootfile, = directory.glob('higgsCombine*.root')
        rows = []
        with uproot.open(rootfile) as handle:
            tree = handle['limit']
            for i,q in enumerate(tree['quantileExpected'].array(library='np')):
                if q > 0:
                    v = tree['limit'].array(entry_start=i,entry_stop=i+1,library='np')[0]
                    rows.append((float(q),float(v)))
        rows.sort()
        assert len(rows) == 5
        report.update(quantiles=[q for q,v in rows],limits=[v for q,v in rows],
            output_root=str(rootfile),output_root_sha256=sha(rootfile))
        jax_result = json.loads(jax_report.read_text())
        report['comparisons'] = {name: [abs(a-b)/abs(b) for (_,a),b in zip(rows,values,strict=True)]
            for name,values in [('dhi_relative_errors',jax_result['reference_limits']),('jax_relative_errors',jax_result['limits'])]}
    save_json(output,report)
    print(report['status'],report.get('limits'),report['wall_seconds'],flush=True)


if __name__ == '__main__':
    main()
