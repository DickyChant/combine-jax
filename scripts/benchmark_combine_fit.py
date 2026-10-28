"""Measure the actual Combine v11 executable with an explicit native fit scope."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

from tensor_benchmark_common import provenance, save_json, sha


def copy_native_covariance(matrix):
    """Copy a native symmetric matrix once, before its RooFitResult is closed."""
    import numpy as np
    size = int(matrix.GetNrows())
    assert int(matrix.GetNcols()) == size
    if not size:
        return np.empty((0, 0), dtype=np.float64)
    view = matrix.GetMatrixArray()
    # cppyy's multidimensional LowLevelView reshape can retain scalar strides.
    # Copy a bounded flat view, then let NumPy form the two-dimensional layout.
    view.reshape((size*size,))
    return np.array(view, dtype=np.float64, copy=True).reshape(size, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('build/tensor_benchmark_inputs/manifest.json'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-limit', type=float, default=10800.)
    parser.add_argument('--minimizer-tolerance', type=float, default=.1)
    parser.add_argument('--minimizer-strategy', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--fit-scope', choices=('migrad', 'migrad_hesse_minos'), default='migrad')
    parser.add_argument('--no-call-counters', action='store_true',help='For instrumentation equivalence checks only')
    parser.add_argument('--preflight-only', action='store_true',help='Validate actual CLI state without a fit')
    parser.add_argument('--fixed-parameter', nargs=2, action='append', default=[], metavar=('NAME','VALUE'),
                        help='Native profile point; repeat for additional fixed coordinates')
    parser.add_argument('--profile-plan',type=Path,help='Shared baseline-bound physical profile grid')
    parser.add_argument('--profile-index',type=int,help='Point index in --profile-plan')
    args = parser.parse_args()
    if (args.profile_plan is None)!=(args.profile_index is None):
        parser.error('--profile-plan and --profile-index must be supplied together')
    if args.profile_plan is not None and args.fixed_parameter:
        parser.error('The shared plan supplies the fixed profile coordinate')
    assert args.wall_limit > 0 and not args.output.exists() and not args.directory.exists()
    assert 0 < args.minimizer_tolerance < float('inf')
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    from combine_jax.tensor.parameter_info import model_parameter_info
    from combine_jax.tensor.fit_policy import check_native_fit_state, mass_cli_arguments, combine_v11_fit_context
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    assert not any('/lcg/views/' in p for p in sys.path) and 'jax' not in sys.modules
    case = json.loads(args.manifest.read_text())[args.case]
    for p, digest in case['hashes'].items():
        assert sha(p) == digest, p
    from native_fit_model import load_native_fit_model
    model,model_load = load_native_fit_model(case['model'])
    from tensor_benchmark_dataset import benchmark_dataset
    dataset_name = benchmark_dataset(case, model, label=args.case)
    profile_plan=None
    if args.profile_plan is not None:
        from tensor_profile_plan import load_plan
        import platform
        plan=load_plan(args.profile_plan)
        assert plan['inputs']['case']==args.case and plan['dataset']==dataset_name
        assert plan['inputs']['manifest']==str(args.manifest)
        assert plan['host']==platform.node() and plan['initial']=='imported nominal'
        assert plan['model_sha256']=={name:sha(Path(case['model'])/name) for name in ('model.json','arrays.npz')}
        assert 0<=args.profile_index<len(plan['grid'])
        args.fixed_parameter=[(plan['parameter'],plan['grid'][args.profile_index])]
        profile_plan=dict(path=str(args.profile_plan),sha256=sha(args.profile_plan),index=args.profile_index)
    profile_fixed={name:float(value) for name,value in args.fixed_parameter}
    if len(profile_fixed)!=len(args.fixed_parameter):
        raise ValueError('Duplicate fixed profile parameter')
    if profile_fixed:
        from combine_jax.tensor.profile_point import configured_profile_model, apply_native_profile_point
        model=configured_profile_model(model,profile_fixed)
    info = model_parameter_info(model)
    names = info.metadata['names']
    prepared = args.directory/'input.root'
    discrete = {c['category']: c['index'] for c in model.metadata['provenance']['discrete_choices'].values()}
    fixed = [n for n, lo, hi in zip(names, info.arrays['lower'], info.arrays['upper'], strict=True) if lo == hi]
    with open_workspace(case['workspace']) as workspace:
        for n, config in case['configuration'].items():
            workspace.var(n).setVal(config['value'])
            workspace.var(n).setConstant(config['constant'])
        if profile_fixed:
            apply_native_profile_point(root,workspace,model)
        for n, v in zip(names, info.arrays['initial'], strict=True):
            workspace.var(n).setVal(float(v))
        for n, v in discrete.items():
            workspace.cat(n).setIndex(v)
        fit_state=check_native_fit_state(workspace,model)
        data=workspace.data(dataset_name)
        dataset_state=dict(name=dataset_name,entries=int(data.numEntries()),sum_weights=float(data.sumEntries()))
        if 'dataset' in model.metadata['provenance']:
            from combine_jax.dataset_identity import workspace_dataset_identity
            dataset_state['identity']=workspace_dataset_identity(workspace,data)
            assert dataset_state['identity']==model.metadata['provenance']['dataset']['identity']
        workspace.writeToFile(str(prepared))
    executable = Path(sys.prefix)/'bin/combine'
    assert executable.is_file()
    command = [sys.executable, str(executable), '-M', 'MultiDimFit', '-d', str(prepared.resolve()),
        '--algo', 'none', '--saveWorkspace' if args.fit_scope == 'migrad' else '--saveFitResult',
        '--cminDefaultMinimizerStrategy', str(args.minimizer_strategy),
        '--cminDefaultMinimizerTolerance', str(args.minimizer_tolerance),
        '--cminDefaultMinimizerType', 'Minuit2', '--cminDefaultMinimizerAlgo', 'Migrad',
        '-v', '3', '-n', '.tensor_benchmark', '-D', dataset_name, *mass_cli_arguments(fit_state)]
    freeze = fixed + list(discrete) + [n for n, c in case['configuration'].items() if c['constant']]
    if freeze:
        command += ['--freezeParameters', ','.join(dict.fromkeys(freeze))]
    report = dict(provenance(), status='running', stage='combine_executable', script_sha256=sha(__file__),
        case=args.case, manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        root_version=str(root.gROOT.GetVersion()), combine_version=version('higgs-combine-tool'),
        input_workspace_sha256=sha(case['workspace']), prepared_workspace=str(prepared),
        stopping_rule=dict(type='Minuit2', algorithm='Migrad', strategy=args.minimizer_strategy,
                           tolerance=args.minimizer_tolerance, error_definition=.5,
                           raw_nll_edm_target=.002*args.minimizer_tolerance*.5),
        prepared_workspace_sha256=sha(prepared), command=command, cwd=str(args.directory.resolve()),
        fit_scope=args.fit_scope, initial_parameters=names, initial_values=info.arrays['initial'], dataset=dataset_name,
        fit_state=fit_state,
        native_fit_model_load=model_load,
        profile_plan=profile_plan,
        profile_point=model.metadata.get('profile_point'),
        scope='Complete executable wall includes startup, input, the selected native fit scope and output. MIGRAD scope uses --algo none --saveWorkspace: no requested post-fit HESSE/MINOS; strategy 1 may compute curvature internally. Parameter errors come from the native saved snapshot. Workspace preparation and canonical postcheck are excluded.',
        discrete_choices=discrete, freeze=freeze, runtime_flags_for_postcheck=REFERENCE_FLAGS,
        wall_limit_seconds=args.wall_limit, rss_limit_mib=11264,
        rss_limit_scope='Current child-process-group RSS plus parent peak RSS; a conservative combined guard.')
    save_json(args.output, report)
    log = args.directory/'combine.log'
    child_environment=dict(os.environ)
    try:
        from native_fit_preflight import run_preflight
        report['stage']='native_state_preflight';save_json(args.output,report)
        run_preflight(command,args.directory/'preflight',model,open_workspace,report,lambda:save_json(args.output,report))
        assert report['preflight']['dataset']==dataset_state
        if args.preflight_only:
            report.update(status='passed_preflight_only',stage='complete')
            return
        if not args.no_call_counters:
            report['stage']='counter_build';save_json(args.output,report)
            from native_minuit_counters import build_library
            library,metadata=build_library(args.directory,root)
            report['call_instrumentation']=metadata
            counter_prefix=(args.directory/'native_calls').resolve()
            child_environment['LD_PRELOAD']=str(library)
            child_environment['COMBINE_JAX_NATIVE_COUNTER_PREFIX']=str(counter_prefix)
        report['stage']='combine_executable';save_json(args.output,report)
        started = time.perf_counter()
        peak = 0.
        combined_peak = 0.
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=args.directory, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True,env=child_environment)
            finished = threading.Event()
            completion = {}
            def reap():
                completion['returncode'] = child.wait()
                completion['time'] = time.perf_counter()
                finished.set()
            waiter = threading.Thread(target=reap, daemon=True)
            waiter.start()
            def interrupted(signum, frame):
                report['interruption_signal'] = signum
                raise InterruptedError(f'Native fit controller received signal {signum}')
            previous_term = signal.signal(signal.SIGTERM, interrupted)
            try:
                last = started
                while not finished.wait(.2):
                    elapsed = time.perf_counter()-started
                    ps = subprocess.run(['ps','--no-headers','-g',str(child.pid),'-o','rss='], capture_output=True, text=True)
                    rss = sum(float(v) for v in ps.stdout.split())/1024
                    peak = max(peak, rss)
                    combined_peak = max(combined_peak, rss+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                    if combined_peak > 11264 or elapsed > args.wall_limit:
                        raise RuntimeError(f'Combine combined resource limit: {combined_peak:.1f} MiB, {elapsed:.1f} s')
                    if time.perf_counter()-last > 20:
                        print('combine heartbeat', elapsed, peak, flush=True)
                        report.update(elapsed_seconds=elapsed, child_process_group_peak_rss_mib=peak,
                                      conservative_combined_peak_rss_mib=combined_peak)
                        save_json(args.output, report)
                        last = time.perf_counter()
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                waiter.join()
        report.update(executable_wall_seconds=completion['time']-started, returncode=child.returncode,
            child_process_group_peak_rss_mib=peak, child_peak_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
            conservative_combined_peak_rss_mib=combined_peak,
            log=str(log), log_sha256=sha(log))
        assert child.returncode == 0
        if args.fit_scope == 'migrad':
            text = log.read_text()
            summaries = re.findall(r'Minuit2Minimizer : (Valid|Invalid) minimum - status = (\d+).*?Edm\s*=\s*([\deE.+-]+).*?Nfcn\s*=\s*(\d+)', text, re.S)
            report['minimizer_summaries'] = [dict(valid=s[0]=='Valid',status=int(s[1]),edm=float(s[2]),function_calls=int(s[3])) for s in summaries]
            # Multiple MIGRAD calls need a separately checked aggregation policy.
            assert len(summaries) == 1, report['minimizer_summaries']
            summary=report['minimizer_summaries'][0]
            assert summary['valid'] and summary['status']==0
            assert 'Running Minos' not in text
            results=list(args.directory.glob('higgsCombine*.root')); assert len(results)==1,results
            with open_workspace(results[0]) as saved:
                snapshot=saved.getSnapshot('MultiDimFit'); assert snapshot
                for name,value in fit_state['fixed_real_values'].items():
                    variable=snapshot.find(name)
                    assert variable and variable.isConstant() and variable.getVal()==value, name
                for name,value in fit_state['category_indices'].items():
                    assert saved.cat(name).getIndex()==value,name
                values=np.array([snapshot.find(n).getVal() for n in names])
                errors=np.array([snapshot.find(n).getError() if n not in fixed else 0. for n in names])
            report['fit']=dict(status=summary['status'],edm=summary['edm'],parameters=names,values=values,errors=errors,
                covariance_quality=None,covariance_scope='Not extracted in MIGRAD-only CLI scope; snapshot errors are uncertified diagnostics at HHVV kinks.')
            report['evaluation_counts']=dict(function=summary['function_calls'],analytic_gradient_callbacks=0,
                numerical_gradient_evaluations=None,scope='Native Minuit2 Nfcn, including calls used by numerical derivatives. Numerical-gradient invocation count is not exposed by the CLI.')
            if not args.no_call_counters:
                from native_minuit_counters import read_counts
                evidence,counts=read_counts(counter_prefix,summary['function_calls'])
                report['call_instrumentation']['capture']=evidence
                report['evaluation_counts']=counts
            logger=args.directory/'combine_logger.out'
            runtime_text=text+(logger.read_text() if logger.exists() else '')
            matches=re.findall(r'Minimized in ([\d.]+) seconds',runtime_text)
            report['migrad_seconds_from_native_logger']=[float(s) for s in matches]
            times=set(report['migrad_seconds_from_native_logger'])
            assert len(times)==1, report['migrad_seconds_from_native_logger']
            report['migrad_seconds']=times.pop()
        else:
            results = list(args.directory.glob('multidimfit*.root'))
        if args.fit_scope != 'migrad':
            read_full_fit(root, results, info, names, fixed, model, report, args, np)
        report['fit']['edm_target_met']=0 <= report['fit']['edm'] < report['stopping_rule']['raw_nll_edm_target']
        with open_workspace(prepared) as workspace:
            with combine_v11_fit_context(workspace):
                for n, v in zip(names, report['fit']['values'], strict=True):
                    workspace.var(n).setVal(float(v))
                config = workspace.obj('ModelConfig')
                nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data(dataset_name), config.GetNuisanceParameters())
                assert not nll.hasGradient(), 'Native counting policy assumes numerical derivatives'
                if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                    nll.setChannelMasks(config.GetPdf().channelMasks())
                nll.setAnalyticBarlowBeeston(True)
                nll.clearConstantZeroPoint(); nll.clearZeroPoint()
                root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
                root.RooAbsReal.clearEvalErrorLog()
                report['fit']['canonical_nll'] = float(nll.getVal())
                report['fit']['roofit_evaluation_errors'] = int(root.RooAbsReal.numEvalErrors())
                nll.setAnalyticBarlowBeeston(False)
                del nll
        report.update(status='measured_fit_pending_agreement', stage='complete', result_sha256=sha(results[0]))
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        save_json(args.output, report)
    print(report['status'], report['executable_wall_seconds'], report['fit']['status'], flush=True)


def read_full_fit(root, results, info, names, fixed, model, report, args, np):
    assert len(results) == 1, results
    handle = root.TFile.Open(str(results[0]))
    fit = handle.Get('fit_mdf')
    assert fit
    pars = fit.floatParsFinal()
    fitted_names = [p.GetName() for p in pars]
    free = [n for n in names if n not in fixed]
    profiled = model.metadata['components'].get('templates', {}).get('profile_names', [])
    assert set(free) <= set(fitted_names) <= set(free) | set(profiled), (fitted_names, free, profiled)
    report['native_covariance_coordinates'] = dict(fitted_parameter_names=fitted_names,
        analytically_profiled_parameters={n: dict(value=float(pars.find(n).getVal()), error=float(pars.find(n).getError()))
            for n in profiled if n in fitted_names},
        comparison='Take the covariance submatrix for the imported parameter vector. The full native HESSE may restore analytically profiled bin nuisances; their marginalization corresponds to the profiled-objective inverse Hessian.')
    values = np.array(info.arrays['initial'], copy=True)
    errors = np.zeros(len(names))
    covariance = np.zeros((len(names), len(names)))
    native_covariance = copy_native_covariance(fit.covarianceMatrix())
    native_indices = {name:i for i,name in enumerate(fitted_names)}
    free_indices = [i for i,name in enumerate(names) if name not in fixed]
    selected = [native_indices[names[i]] for i in free_indices]
    covariance[np.ix_(free_indices, free_indices)] = native_covariance[np.ix_(selected, selected)]
    for i, n in enumerate(names):
        if n in fixed:
            continue
        values[i] = pars.find(n).getVal()
        errors[i] = pars.find(n).getError()
    report['fit'] = dict(status=int(fit.status()), covariance_quality=int(fit.covQual()), edm=float(fit.edm()),
        min_nll_internal=float(fit.minNll()), parameters=names, values=values, errors=errors, covariance=covariance)
    if len(names) > 64:
        # Keep full HHVV matrices in numeric artifacts, not millions of JSON
        # entries. The small-card schema stays compatible with prior audits.
        numeric = args.output.with_suffix('.npz')
        assert not numeric.exists()
        np.savez_compressed(numeric, parameter_names=np.array(names), values=values,
                            errors=errors, covariance=covariance)
        del report['fit']['covariance']
        report['fit']['covariance_arrays'] = str(numeric)
        report['fit']['covariance_arrays_sha256'] = sha(numeric)
    handle.Close()

if __name__ == '__main__':
    main()
