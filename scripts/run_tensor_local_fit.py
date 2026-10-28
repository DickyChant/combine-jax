"""Supervise a local CPU fit and a separate native same-point check.

Launch this controller detached for long runs. Each child has its own process
group; exit codes and source hashes are saved even when a child fails. A saved
fit remains subject to independent convergence and covariance acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tensor_benchmark_common import provenance, save_json, sha


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='hhvv')
    parser.add_argument('--fit-max-iterations', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True, help='Fit benchmark JSON path')
    args = parser.parse_args()
    assert args.fit_max_iterations > 0
    root = Path.cwd()
    assert Path(sys.executable).parent == root/'.venv/bin', 'Use the project venv'
    assert args.output.parent == Path('reports') and args.output.suffix == '.json'
    manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    case = manifest[args.case]
    driver = args.output.with_name(args.output.stem+'_driver.json')
    native = args.output.with_name(args.output.stem+'_native_point.json')
    for path in (args.output, driver, native, args.output.with_suffix('.npz'),
                 args.output.with_name(args.output.stem+'_first_fit.npz')):
        assert not path.exists(), path
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    paths = [*sorted(Path('src/combine_jax').rglob('*.py')), Path(__file__),
             *map(Path, ('scripts/benchmark_tensor.py', 'scripts/tensor_benchmark_common.py',
                         'scripts/check_tensor_fit_candidate.py', 'scripts/check_exported_fit_candidate.py')),
             manifest_path, Path(case['workspace']),
             *[Path(case['model'])/name for name in ('model.json', 'arrays.npz')]]
    frozen = {str(path): sha(path) for path in paths}
    report = dict(provenance(), status='running', case=args.case, controller=identity(os.getpid()),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  frozen_inputs=frozen, fit_output=str(args.output), native_output=str(native), stages=[],
                  scope='Sequential local CPU benchmark and isolated Combine v11 same-point check. '
                        'No convergence or uncertainty acceptance follows from child exit status. '
                        'The benchmark saves its first fit before any warm repeat; its 11 GiB RSS guard remains active.')
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(label, command, output):
        nonlocal child
        log = output.with_suffix('.log')
        row = dict(stage=label, command=command, output=str(output), log=str(log),
                   status='running', started_unix=time.time())
        report['stages'].append(row); save_json(driver, report)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            row['child'] = identity(child.pid); save_json(driver, report)
            child.wait()
        row.update(returncode=child.returncode, finished_unix=time.time(), log_sha256=sha(log),
                   status='completed' if child.returncode == 0 else 'failed')
        if output.exists():
            row.update(output_sha256=sha(output), output_status=json.loads(output.read_text())['status'])
        save_json(driver, report)
        assert child.returncode == 0, row

    save_json(driver, report)
    try:
        run('fit', [sys.executable, 'scripts/benchmark_tensor.py', '--model', case['model'],
                    '--backend', 'cpu', '--operation', 'fit', '--fit-max-iterations',
                    str(args.fit_max_iterations), '--batch-size', '4', '--repeat', '1',
                    '--output', str(args.output)], args.output)
        fit = json.loads(args.output.read_text())
        assert fit['status'] == 'measured_pending_validation'
        checkpoint = fit['first_fit_checkpoint']
        assert sha(checkpoint['arrays']) == checkpoint['arrays_sha256']
        report['first_fit_checkpoint'] = checkpoint
        run('native_same_point', [sys.executable, 'scripts/check_tensor_fit_candidate.py',
                                 '--fit', str(args.output), '--case', args.case,
                                 '--output', str(native)], native)
        assert all(sha(path) == digest for path, digest in frozen.items()), 'Frozen source/input changed'
        report.update(status='completed_pending_fit_audit', frozen_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            report['interrupted_child_returncode'] = child.returncode
        report['finished_unix'] = time.time(); save_json(driver, report)


if __name__ == '__main__':
    main()
