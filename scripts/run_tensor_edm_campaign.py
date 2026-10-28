"""Sequential CPU/GPU/native comparison under the accepted default EDM rule."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from tensor_benchmark_common import save_json,sha


def selected_backends(runtime, requested):
    backends=tuple(requested) if requested is not None else (('cpu',) if runtime=='local' else ('gpu','cpu'))
    if not backends or len(set(backends))!=len(backends):
        raise ValueError('Select each backend at most once')
    if any(b not in ('cpu','gpu') for b in backends) or (runtime=='local' and backends!=('cpu',)):
        raise ValueError('The local runtime supports only CPU; GPU requires LCG')
    return backends


def check_hhvv_native_state(native, metadata, *, manifest_sha256, workspace_sha256, returncode):
    """Require the actual native CLI to verify the imported Asimov fit problem."""
    assert native['case']=='hhvv' and native['dataset']=='asimov_prefit'
    assert not native.get('interruption_signal') and returncode>=0, 'Cancellation must not start more measurements'
    assert 'KeyboardInterrupt' not in native.get('exception',''), 'Cancellation must not start more measurements'
    assert native['manifest_sha256']==manifest_sha256
    assert native['input_workspace_sha256']==workspace_sha256
    policy=metadata['parameter_info']['fit_policy']
    assert policy['policy']=='combine_v11_default'
    assert native['preflight']['status']=='passed' and native['preflight']['returncode']==0
    assert native['preflight']['initial_state']==native['fit_state']==policy
    dataset=metadata['provenance']['dataset']
    assert dataset['name']=='asimov_prefit'
    assert native['preflight']['dataset']['name']==dataset['name']
    assert native['preflight']['dataset']['identity']==dataset['identity']
    assert native['root_version']=='6.40.00' and native['combine_version']=='11.0.0'
    assert native['fit_scope']=='migrad'
    assert native['stopping_rule']==dict(type='Minuit2',algorithm='Migrad',strategy=1,
        tolerance=.1,error_definition=.5,raw_nll_edm_target=1e-4)


def hhvv_native_outcome(native, metadata, *, manifest_sha256, workspace_sha256, returncode):
    """Keep independent tensor evidence after a failed minimization, not cancellation."""
    assert native['status'] in ('failed','measured_fit_pending_agreement')
    check_hhvv_native_state(native,metadata,manifest_sha256=manifest_sha256,
        workspace_sha256=workspace_sha256,returncode=returncode)
    fit=native.get('fit',{})
    eligible=(returncode==0 and native['status']=='measured_fit_pending_agreement'
        and fit.get('status')==0 and fit.get('edm_target_met') is True
        and math.isfinite(fit.get('edm',math.nan)) and 0<=fit['edm']<1e-4
        and fit.get('roofit_evaluation_errors')==0)
    return dict(native_state_gate='passed',native_status=native['status'],native_returncode=returncode,
        minimum_comparison_eligible=bool(eligible),
        scope='A passed native CLI preflight permits independent tensor fit/Hessian measurements after a terminal native fit failure. Minimum agreement and timing ratios still require a valid native fit and the unchanged numerical audits. No retry or stopping-rule relaxation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default='gpu_tensor_edm_')
    parser.add_argument('--gpu-class',choices=('primary','t4'),default='primary')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--runtime',choices=('lcg','local'),default='lcg')
    parser.add_argument('--hhvv-fit-order',choices=('native-first','tensor-first'),default='native-first',
                        help='Tensor-first verifies the native CLI start before fitting; nominal Hessians follow the native fit')
    parser.add_argument('--gpu-hessian-batch-size',type=int,
                        help='Exact AD Hessian column batch size for HHVV GPU fits and Hessians')
    parser.add_argument('--backends',nargs='+',choices=('cpu','gpu'),
                        help='Run only these backends, in order; LCG still verifies its GPU first')
    args=parser.parse_args(); assert re.fullmatch(r'gpu_tensor_[a-z0-9_]+_',args.prefix)
    if args.gpu_hessian_batch_size is not None and args.gpu_hessian_batch_size<1:
        parser.error('gpu-hessian-batch-size must be positive')
    try:backends=selected_backends(args.runtime,args.backends)
    except ValueError as exc:parser.error(str(exc))
    if args.runtime=='local':
        assert Path(sys.executable).parent==Path.cwd()/'.venv/bin', 'Use the project venv'
    output=Path('reports')/(args.prefix+'driver.json'); assert not output.exists()
    runtime_path=Path('build/edm_runtime_manifest.json');runtime=json.loads(runtime_path.read_text())
    assert all(sha(p)==digest for p,digest in runtime['sources'].items())
    manifest_path=Path('build/tensor_benchmark_inputs/manifest.json')
    manifest=json.loads(manifest_path.read_text())
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    report=dict(status='running',script_sha256=sha(__file__),source_revision=runtime['source_revision'],
        runtime_manifest_sha256=sha(runtime_path),manifest_sha256=sha(manifest_path),started_unix=time.time(),commands=[],
        criterion=dict(native='Minuit2/Migrad, strategy 1, tolerance 0.1, NLL Up=0.5',raw_edm_target=1e-4,
            tensor='Device BFGS corrected estimated EDM. Exact curvature refines smooth smoke fits; HHVV exact curvature is diagnostic under accepted clipping policy.'),
        runtime=args.runtime,backends=backends,hhvv_fit_order=args.hhvv_fit_order,
        scope='Sequential project workloads on one assigned node, at most eight CPUs. GPU checked before model work when using LCG. Other users are uncontrolled. Convergence and successful-fit speed require independent audits. T4 is secondary; its approximately 1/32 FP64/FP32 ratio is user hardware context, not a measured workload ratio.')
    child=None
    def path(label):return str(Path('reports')/(args.prefix+label+'.json'))
    def run(command,label,required=True):
        nonlocal child
        log=Path('reports')/(args.prefix+label+'.log')
        row=dict(command=command,log=str(log),status='running',started_unix=time.time())
        report['commands'].append(row);save_json(output,report)
        print('start',label,flush=True)
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=child.pid;save_json(output,report);child.wait()
        row.update(returncode=child.returncode,finished_unix=time.time(),log_sha256=sha(log),
                   status='completed' if child.returncode==0 else 'failed')
        save_json(output,report)
        if required:assert child.returncode==0,row
        return child.returncode
    def lcg(script,*items):
        prefix=[sys.executable] if args.runtime=='local' else ['bash','scripts/run_tensor_lcg.sh']
        return [*prefix,script,*map(str,items)]
    def native(script,*items):
        prefix=[sys.executable] if args.runtime=='local' else ['bash','scripts/run_tensor_native.sh']
        return [*prefix,script,*map(str,items)]
    def interrupted(signum,frame):raise InterruptedError(f'Campaign received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save_json(output,report)
    try:
        if args.runtime=='lcg':
            run(lcg('scripts/verify_tensor_gpu.py','--output',path('environment')),'environment')
            environment=json.loads(Path(path('environment')).read_text())
            kind=environment['devices'][0]['kind']
            assert ('T4' in kind) if args.gpu_class=='t4' else ('A100' in kind or 'H100' in kind)
        policies={};metadata_by_case={}
        for case,entry in manifest.items():
            doc=json.loads((Path(entry['model'])/'model.json').read_text())
            metadata_by_case[case]=doc['metadata']
            if not args.smoke_only or case in ('counting','automcstats'):
                from tensor_benchmark_dataset import benchmark_dataset
                benchmark_dataset(entry,SimpleNamespace(metadata=doc['metadata']),label=case)
            policies[case]=doc['metadata']['parameter_info'].get('fit_policy')
            assert policies[case] and policies[case]['policy']=='combine_v11_default',case
        gpu_batch=args.gpu_hessian_batch_size or (1 if args.gpu_class=='t4' else 4)
        report['hhvv_gpu_hessian_batch_size']=gpu_batch
        gpu_memory=8 if args.gpu_class=='t4' else 28
        for case in ('counting','automcstats'):
            run(native('scripts/benchmark_combine_fit.py','--case',case,'--directory','build/edm_native_'+case,
                       '--output',path(case+'_native')),case+'_native')
            for backend in backends:
                label=case+'_'+backend+'_fit'
                run(lcg('scripts/benchmark_tensor.py','--model',manifest[case]['model'],'--backend',backend,
                    '--operation','fit','--fit-stopping-criterion','edm','--fit-max-iterations','2000',
                    '--single-fit','--repeat','1','--output',path(label)),label)
                run(native('scripts/check_tensor_fit_candidate.py','--fit',path(label),'--case',case,
                    '--output',path(label+'_native_point')),label+'_native_point')
                run(lcg('scripts/audit_tensor_edm_comparison.py','--case',case,'--tensor',path(label),
                    '--native',path(case+'_native'),'--native-point',path(label+'_native_point'),
                    '--output',path(label+'_audit')),label+'_audit')
        if args.smoke_only:
            assert all(sha(p)==digest for p,digest in runtime['sources'].items())
            report['status']='smoke_checks_passed'
            return
        model=manifest['hhvv']
        report['hhvv_reference_validation']=model.get('reference_validation',{})
        save_json(output,report)
        for backend in backends:
            captures=[]
            for operation in ('nll','value_gradient'):
                label=backend+'_'+operation;captures.append(path(label))
                run(lcg('scripts/benchmark_tensor.py','--model',model['model'],'--points',model['points'],
                    '--backend',backend,'--operation',operation,'--repeat','26',
                    '--compiler-memory-limit-gib',gpu_memory if backend=='gpu' else 8,
                    '--output',path(label)),label)
            run(lcg('scripts/audit_tensor_benchmarks.py','--case','hhvv','--reports',*captures,
                '--output',path(backend+'_values_audit')),backend+'_values_audit')
        run(native('scripts/benchmark_native_tensor.py','--case','hhvv','--repeat','26',
                   '--output',path('native_nll')),'native_nll')
        state_hashes=dict(manifest_sha256=sha(manifest_path),workspace_sha256=sha(model['workspace']))
        def native_fit():
            native_code=run(native('scripts/benchmark_combine_fit.py','--case','hhvv','--directory','build/edm_native_hhvv',
                       '--wall-limit','21600','--output',path('native_fit')),'native_fit',required=False)
            native_result=json.loads(Path(path('native_fit')).read_text())
            outcome=hhvv_native_outcome(native_result,metadata_by_case['hhvv'],**state_hashes,returncode=native_code)
            report.update(hhvv_native_state_gate=outcome['native_state_gate'],hhvv_native_fit_outcome=outcome)
            save_json(output,report)
            return outcome
        measured_fits=set()
        def tensor_operation(backend,operation):
            batch=gpu_batch if backend=='gpu' else 1
            memory=gpu_memory if backend=='gpu' else 8
            label=backend+'_'+operation
            extra=[] if operation=='hessian' else ['--fit-stopping-criterion','edm','--fit-edm-validation','estimated',
                '--fit-uncertainty-note','Four accepted native HHVV background clipping bins; covariance and uncertainties are not certified.',
                '--fit-max-iterations','10000','--single-fit']
            run(lcg('scripts/benchmark_tensor.py','--model',model['model'],'--backend',backend,
                '--operation',operation,'--batch-size',batch,'--compiler-memory-limit-gib',memory,
                '--repeat','1',*extra,'--output',path(label)),label,required=False)
            if operation=='fit' and Path(path(label)).exists():
                measured=json.loads(Path(path(label)).read_text())
                if measured.get('status')=='measured_pending_validation':
                    measured_fits.add(backend)
                    run(native('scripts/check_tensor_fit_candidate.py','--fit',path(label),
                        '--output',path(label+'_native_point')),label+'_native_point',required=False)
        def compare_fit(backend,outcome):
            if backend in measured_fits and outcome['minimum_comparison_eligible']:
                label=backend+'_fit'
                run(lcg('scripts/audit_tensor_edm_comparison.py','--case','hhvv','--tensor',path(label),
                    '--native',path('native_fit'),'--native-point',path(label+'_native_point'),
                    '--allow-uncertified-curvature','--output',path(label+'_comparison')),label+'_comparison',required=False)
        if args.hhvv_fit_order=='tensor-first':
            code=run(native('scripts/benchmark_combine_fit.py','--case','hhvv','--preflight-only',
                '--directory','build/edm_native_hhvv_preflight','--output',path('native_preflight')),'native_preflight')
            preflight=json.loads(Path(path('native_preflight')).read_text())
            assert preflight['status']=='passed_preflight_only'
            check_hhvv_native_state(preflight,metadata_by_case['hhvv'],**state_hashes,returncode=code)
            report['hhvv_native_state_gate']='passed';save_json(output,report)
            for backend in backends:tensor_operation(backend,'fit')
            outcome=native_fit()
            for backend in backends:compare_fit(backend,outcome)
            for backend in backends:tensor_operation(backend,'hessian')
        else:
            outcome=native_fit()
            for backend in backends:
                tensor_operation(backend,'fit')
                compare_fit(backend,outcome)
                tensor_operation(backend,'hessian')
        assert all(sha(p)==digest for p,digest in runtime['sources'].items())
        complete=outcome['minimum_comparison_eligible'] and all(r['returncode']==0 for r in report['commands'])
        report['status']='completed_pending_audit' if complete else 'partial_failures_pending_audit'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            report['interrupted_child_returncode']=child.returncode
        report['finished_unix']=time.time();save_json(output,report)


if __name__=='__main__':main()
