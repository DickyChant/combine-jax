"""Sequence independent native ROOT, importer ROOT and CUDA processes."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

from .options import request
from .resources import save_json, sha


def stage(command, environment, log, *, rss_limit_mib=11264):
    """Bound aggregate process-group RSS, preserving logs on any failure."""
    started = time.perf_counter()
    peak = 0
    with log.open('w') as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                rss = 0
                for item in Path('/proc').glob('[0-9]*/stat'):
                    try:
                        fields = item.read_text().rsplit(')',1)[1].split()
                        if int(fields[2]) == process.pid:
                            rss += int(fields[21])*os.sysconf('SC_PAGE_SIZE')/2**20
                    except (OSError,ValueError,IndexError):
                        pass
                peak = max(peak,rss)
                if rss > rss_limit_mib:
                    raise MemoryError(f'Process group RSS {rss:.1f} MiB exceeds {rss_limit_mib} MiB')
                time.sleep(.5)
            if process.returncode:
                tail = '\n'.join(log.read_text(errors='replace').splitlines()[-20:])
                raise RuntimeError(f'Stage exited {process.returncode}; log {log}\n{tail}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            raise
    return dict(wall_seconds=time.perf_counter()-started, sampled_group_peak_rss_mib=peak,
                rss_limit_mib=rss_limit_mib, log=str(log), log_sha256=sha(log))


def run_limits(args):
    started = time.perf_counter()
    if socket.gethostname().split('.')[0] == 'lxplus905':
        raise RuntimeError('Execution on lxplus905 is prohibited')
    config = request(args)
    from ..methods.asymptotic import LimitOptions
    LimitOptions(confidence_level=config['confidence_level'],
        r_abs_accuracy=config['r_abs_accuracy'],r_rel_accuracy=config['r_rel_accuracy'])
    prefix = args.runtime or os.environ.get('DHI_COMBINE_JAX')
    if not prefix:
        raise ValueError('Set DHI_COMBINE_JAX or use --runtime with scripts/setup_runtime.sh output')
    prefix = Path(prefix).resolve()
    runtime = json.loads((prefix/'runtime.json').read_text())
    config.update(native_version=runtime['native_version'],
                  native_library_sha256=runtime['native_library_sha256'])
    for name, checksum in runtime['files'].items():
        if sha(prefix/name) != checksum:
            raise ValueError('Runtime source checksum mismatch: '+name)
    native = os.environ.get('COMBINE_JAX_NATIVE_CMSSW',runtime['native_cmssw'])
    module = prefix/'code/src/combine_jax/runtime'
    output = (args.output or Path(f'higgsCombine{args.name}.AsymptoticLimits.mH{args.mass:g}.{args.seed}.root')).resolve()
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Output or JSON sidecar already exists')
    if args.workdir:
        workdir = args.workdir.resolve()
        workdir.mkdir(parents=True,exist_ok=False)
    else:
        workdir = Path(tempfile.mkdtemp(prefix='combine-jax-',dir=Path.cwd()))
    config_path = workdir/'request.json'
    save_json(config_path,config)
    report = dict(status='running',request=config,runtime=runtime,workdir=str(workdir),stages={})
    output.parent.mkdir(parents=True,exist_ok=True)
    save_json(sidecar,report)
    environment = dict(os.environ)
    environment.update(JAX_ENABLE_X64='true',JAX_PLATFORMS='cuda' if args.backend=='gpu' else 'cpu',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false')
    lcg = ['bash',str(module/'run_lcg.sh'),str(prefix),runtime['lcg_view']]
    def run(name,command,env):
        print('Starting',name,flush=True)
        report['stages'][name] = stage(command,env,workdir/(name+'.log'))
        save_json(sidecar,report)
    try:
        # Fail a bad CUDA allocation before spending time generating/importing.
        run('device_check',lcg+['-m','combine_jax.runtime.fit','--verify',args.backend],environment)
        run('native_asimov',['bash',str(module/'run_native.sh'),native,
            str(module/'native_asimov.py'),str(config_path),str(workdir/'native')],environment)
        importer_env = {k:v for k,v in environment.items() if k in
            ('HOME','USER','TMPDIR','LANG','CUDA_VISIBLE_DEVICES')}
        importer_env.update(PATH=str(prefix/'venv-import/bin')+':/usr/bin:/bin',
            PYTHONPATH=str(prefix/'code/src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
        run('import',[str(prefix/'venv-import/bin/python'),'-m','combine_jax.runtime.import_model',
            str(config_path),str(workdir/'native/generation.json'),str(workdir/'import')],importer_env)
        run('fit',lcg+['-m','combine_jax.runtime.fit',str(config_path),
            str(workdir/'import/preparation.json'),str(workdir/'fit.json')],environment)
        fit = json.loads((workdir/'fit.json').read_text())
        if not fit['all_fits_converged']:
            raise RuntimeError('Failed profile fit; refusing a successful ROOT output')
        from ..methods.root_output import write_limit_tree
        elapsed = time.perf_counter()-started
        write_limit_tree(output,fit['rows'],mass=args.mass,seed=args.seed,wall_seconds=elapsed)
        report.update(status=fit['status'],fit=fit,output=str(output),output_sha256=sha(output),
            total_wall_seconds=elapsed,setup_wall_seconds=sum(report['stages'][n]['wall_seconds']
                for n in ('device_check','native_asimov','import')),
            receipt=dict(expected_only=True,root_sha256=sha(output),
                         source_workspace_sha256=sha(config['workspace'])))
        print('Expected quantiles:',[(r['quantile'],r['limit'],r['status']) for r in fit['rows']],flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),total_wall_seconds=time.perf_counter()-started)
        # A batch worker's temporary directory disappears after failure. Keep
        # the last fit checkpoint in the transferred receipt, including partial
        # profile convergence and memory measurements. Never replace the primary
        # exception if termination interrupted an evidence write.
        checkpoint = workdir/'fit.json'
        if checkpoint.exists():
            try:
                report['fit'] = json.loads(checkpoint.read_text())
            except (OSError, ValueError) as evidence_error:
                report['fit_checkpoint_error'] = repr(evidence_error)
        raise
    finally:
        save_json(sidecar,report)
