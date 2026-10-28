"""Run separate ROOT/JAX stages under one job, with a process-group RSS guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from hhvv_mx3000_authorization import authorize, sha


def stage_environment(*, gpu=False):
    """Separate the CPU-only ROOT importer from the CUDA-view fit processes."""
    env = {k:v for k,v in os.environ.items() if k not in
           ('PYTHONHOME','PYTHONPATH','ROOTSYS','ROOT_INCLUDE_PATH','LD_LIBRARY_PATH',
            'JAX_PLATFORM_NAME')}
    env.update(PYTHONPATH=str(Path('src').resolve())+':'+str(Path('scripts').resolve()),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', JAX_ENABLE_X64='true',
               JAX_PLATFORMS='cuda' if gpu else 'cpu',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    # Serial GPU code generation bounds concurrent compiler subprocess memory.
    # The LCG wrapper opts in only for this guarded campaign.
    env['COMBINE_JAX_GPU_SERIAL_COMPILE'] = '1' if gpu else '0'
    return env


def main():
    manifest_path = Path('build/hhvv_observed_runtime.json')
    manifest = json.loads(manifest_path.read_text())
    authorize(manifest, 'build/hhvv_observed/input/workspace.root', 'build/hhvv_observed/input/card.txt')
    output = Path('reports/gpu_hhvv_MX3000_observed_driver.json')
    assert not output.exists()
    report = dict(status='running', mass=3000, run='observed', manifest_sha256=sha(manifest_path),
                  stages=[], rss_limit_mib=11264, optional_expected_quantile=.025)
    started = time.monotonic()
    def save():
        report['wall_seconds'] = time.monotonic()-started
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n'); temp.replace(output)
    save()
    def run(label, command, timeout, optional=False, gpu=False):
        env = stage_environment(gpu=gpu)
        row = dict(label=label, command=command, status='running', peak_process_group_rss_mib=0.,
                   jax_platforms=env['JAX_PLATFORMS'], jax_enable_x64=env['JAX_ENABLE_X64'])
        report['stages'].append(row); save()
        then = time.monotonic(); child = None
        try:
            with Path('reports/gpu_hhvv_MX3000_'+label+'.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    ps = subprocess.run(['ps','--no-headers','-g',str(child.pid),'-o','pid=,ppid=,rss=,comm='], capture_output=True, text=True)
                    assert ps.returncode in (0,1)
                    processes = [dict(pid=int(v[0]), ppid=int(v[1]), rss_mib=float(v[2])/1024, command=v[3])
                                 for line in ps.stdout.splitlines() if len(v := line.split(maxsplit=3)) == 4]
                    rss = sum(p['rss_mib'] for p in processes)
                    if rss > row['peak_process_group_rss_mib']:
                        row.update(peak_process_group_rss_mib=rss, processes_at_peak=processes)
                    row['wall_seconds'] = time.monotonic()-then
                    save()
                    if rss > 11264: raise MemoryError('Stage exceeds 11 GiB process-group RSS guard')
                    if row['wall_seconds'] > timeout: raise TimeoutError('Stage time budget exceeded')
                    time.sleep(.5)
            row.update(returncode=child.returncode, status='completed' if child.returncode == 0 else 'failed')
            if child.returncode: raise RuntimeError('Stage '+label+' failed')
        except BaseException as exc:
            row.update(status='failed_or_guarded', error=repr(exc))
            if not optional: raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
            row.update(wall_seconds=time.monotonic()-then, returncode=None if child is None else child.returncode)
            save()
    def gpu(phase):
        return ['bash','scripts/run_tensor_lcg.sh','scripts/run_hhvv_mx3000_observed_gpu.py','--phase',phase]
    try:
        run('observed_import', ['.venv-import/bin/python','scripts/import_hhvv_mx3000_observed.py','--kind','observed'], 7200)
        run('background_fit', gpu('background'), 21600, gpu=True)
        run('conditional_generation', ['bash','scripts/run_native_v10_cmssw.sh','scripts/generate_hhvv_mx3000_conditional_asimov.py'], 1800)
        run('conditional_import', ['.venv-import/bin/python','scripts/import_hhvv_mx3000_observed.py','--kind','conditional_asimov'], 7200)
        run('observed_limit', gpu('observed'), 43200, gpu=True)
        report.update(status='observed_completed', observed_ready_wall_seconds=time.monotonic()-started)
        save()
        # The observed artifact is already complete. This bounded optional stage
        # cannot invalidate it or hold it up for an unbounded expected campaign.
        run('expected025_retry', gpu('expected025'), 600, optional=True, gpu=True)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
