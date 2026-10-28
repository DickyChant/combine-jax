"""Run every tensor test module sequentially in an isolated source worktree."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


def tree_rss_mib(root):
    processes = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            processes[int(entry.name)] = (int(fields[1]), int(fields[21]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {root}
    while True:
        added = {pid for pid, (parent, _) in processes.items() if parent in descendants}
        if added <= descendants:
            break
        descendants.update(added)
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)*os.sysconf('SC_PAGE_SIZE')/1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path,
                        help='Reuse completed module logs only if every recorded source hash still matches')
    args = parser.parse_args()
    worktree = args.worktree.resolve(); output = args.output.resolve()
    assert not output.exists()
    log_directory = output.with_suffix(''); log_directory.mkdir()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=worktree, text=True).strip()
    tests = sorted(worktree.glob('tests/test_tensor_*.py'))
    assert tests
    sources = [*sorted(worktree.glob('src/combine_jax/tensor/*.py')), *tests,
        worktree/'scripts/benchmark_tensor.py', worktree/'scripts/tensor_benchmark_common.py',
        worktree/'tests/conftest.py', worktree/'pyproject.toml']
    hashes = {str(p.relative_to(worktree)): sha(p) for p in sources}
    affinity = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, affinity)
    report = dict(status='running', script_sha256=sha(__file__),
        worktree=str(worktree), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip(),
        controller_pid=os.getpid(),
        controller_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        source_hashes=hashes, test_modules=[str(p.relative_to(worktree)) for p in tests],
        python=sys.executable, cpu_affinity=affinity, tree_rss_limit_mib=4096,
        per_module_wall_limit_seconds=900, started_unix=time.time(), modules=[],
        scope='Complete tensor test-module selection in fresh sequential processes on the isolated '
              'cache/checkpoint branch. Process-tree RSS guard preserves room for the live local fit. '
              'Correctness validation only; shared-host timings are not performance measurements. '
              'This does not rerun the unchanged legacy backend suite or a single-session tensor suite.')
    completed = set()
    if args.resume:
        previous = json.loads(args.resume.read_text())
        for key in ('worktree', 'source_commit', 'source_hashes', 'test_modules', 'python', 'cpu_affinity'):
            assert previous[key] == report[key], key
        assert len({m['module'] for m in previous['modules']}) == len(previous['modules'])
        for row in previous['modules']:
            if row['status'] == 'passed':
                assert row['returncode'] == 0 and 'guard_reason' not in row
                assert sha(row['log']) == row['log_sha256']
                report['modules'].append({**row, 'reused_evidence_from':str(args.resume)})
                completed.add(row['module'])
        report.update(resumed_from=str(args.resume), resumed_from_sha256=sha(args.resume))
    save_json(output, report)
    environment = {**os.environ, 'PYTHONPATH': str(worktree/'src')+os.pathsep+str(worktree/'scripts'),
        'JAX_PLATFORMS':'cpu', 'JAX_ENABLE_X64':'true', 'OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
        'XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false'}
    (worktree/'build/tensor_release_pytest').mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        for test in tests:
            relative = str(test.relative_to(worktree))
            if relative in completed:
                continue
            log = log_directory/(test.stem+'.log')
            command = [sys.executable, '-m', 'pytest', '-q', relative,
                       '--basetemp=build/tensor_release_pytest/'+test.stem]
            row = dict(module=relative, command=command, log=str(log), status='running',
                       peak_tree_rss_mib=0., started_unix=time.time())
            report['modules'].append(row); save_json(output, report)
            print('start', relative, flush=True)
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=worktree, env=environment, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                row.update(pid=child.pid, start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]))
                save_json(output, report)
                started = time.monotonic()
                while child.poll() is None:
                    rss = tree_rss_mib(child.pid)
                    row['peak_tree_rss_mib'] = max(row['peak_tree_rss_mib'], rss)
                    if rss > report['tree_rss_limit_mib'] or time.monotonic()-started > report['per_module_wall_limit_seconds']:
                        row['guard_reason'] = 'process_tree_rss' if rss > report['tree_rss_limit_mib'] else 'wall_time'
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                        break
                    time.sleep(.25)
            row.update(returncode=child.returncode, finished_unix=time.time(), log_sha256=sha(log),
                       status='passed' if child.returncode == 0 and 'guard_reason' not in row else 'failed')
            save_json(output, report)
            print(row['status'], relative, row['peak_tree_rss_mib'], flush=True)
            if row['status'] != 'passed':
                report['status'] = 'failed'; break
        else:
            assert all(sha(worktree/path) == digest for path, digest in hashes.items())
            report['status'] = 'passed_all_tensor_modules'
    except BaseException as exc:
        report.update(status='error', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            report['interrupted_child_returncode'] = child.returncode
        report['finished_unix'] = time.time(); save_json(output, report)
    if report['status'] != 'passed_all_tensor_modules':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
