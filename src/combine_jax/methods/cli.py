"""Expected-only statistical-method command line."""
import json
import os
from pathlib import Path
import platform
import resource
import time


def add_limit_command(commands):
    parser = commands.add_parser('limit', help='AsymptoticLimits expected CLs upper limits')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--workspace', type=Path)
    source.add_argument('--model', type=Path, help='Previously prepared neutral background-Asimov model')
    parser.add_argument('--datacard', '--card', type=Path)
    parser.add_argument('--poi')
    parser.add_argument('--run', choices=('blind', 'expected'),
                        help='Workspace default: blind. Prepared model default: its recorded Asimov mode.')
    parser.add_argument('--noFitAsimov', action='store_true')
    parser.add_argument('--data')
    parser.add_argument('--asimov-report', type=Path, help='Required proof for HHVV conditional expected input')
    parser.add_argument('--rMax', type=float)
    parser.add_argument('--cl', type=float, default=.95)
    parser.add_argument('--rAbsAcc', type=float, default=.0005)
    parser.add_argument('--rRelAcc', type=float, default=.005)
    parser.add_argument('--strictBounds', action='store_true')
    parser.add_argument('--setParameters', default='')
    parser.add_argument('--freezeParameters', default='')
    parser.add_argument('--mass', type=float)
    parser.add_argument('--template-convention',choices=('v10_density','native'),
                        help='Workspace import convention (default: v10_density); native preserves installed-stack behavior')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--output', type=Path, required=True)


def run_limit(args):
    if args.output.exists():
        raise FileExistsError('Preserve existing limit evidence')
    if args.workspace and not args.datacard:
        raise ValueError('--workspace requires --datacard for the neutral importer')
    if args.model and any((args.datacard, args.data, args.asimov_report, args.setParameters,
                           args.freezeParameters, args.mass is not None, args.workdir,args.template_convention)):
        raise ValueError('Workspace preparation options cannot change a prepared --model; prepare a new model')
    if args.backend == 'gpu' and (args.workspace or not os.environ.get('_CONDOR_SCRATCH_DIR')):
        raise ValueError('GPU limits require a prepared --model inside an authorized HTCondor A100/H100 job')
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jaxlib
    assert jax.config.x64_enabled
    devices = jax.devices()
    assert devices and all(d.platform == args.backend for d in devices)
    if args.backend == 'gpu':
        assert len(devices) == 1 and any(x in devices[0].device_kind for x in ('A100','H100'))
    start = time.perf_counter()
    report = dict(status='running', method='AsymptoticLimits', host=platform.node(),
        backend=args.backend, jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        devices=[dict(kind=d.device_kind, platform=d.platform) for d in devices], float64=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .workspace import prepare_expected_workspace, digest
        from .runner import run_expected_model
        from .asymptotic import LimitOptions
        if args.workspace:
            parameters = dict(part.split('=', 1) for part in args.setParameters.split(',') if part)
            model, preparation = prepare_expected_workspace(args.workspace, args.datacard,
                args.workdir or args.output.with_suffix('.work'), poi=args.poi, run=args.run or 'blind',
                no_fit_asimov=args.noFitAsimov, data=args.data or 'data_obs', asimov_report=args.asimov_report,
                r_max=args.rMax, parameters=parameters,
                freeze=[p for p in args.freezeParameters.split(',') if p], mass=125. if args.mass is None else args.mass,
                template_convention=args.template_convention or 'v10_density')
            model_path = Path(preparation['model'])
            report['preparation'] = preparation
        else:
            from ..tensor.bundle import TensorBundle
            model_path = args.model
            model = TensorBundle.load(model_path)
        poi = args.poi or model.metadata.get('asymptotic_asimov', {}).get('poi')
        from .contracts import validate_expected_model
        validate_expected_model(model, poi, run=args.run, no_fit_asimov=args.noFitAsimov)
        result = run_expected_model(model, poi, r_max=args.rMax,
            options=LimitOptions(confidence_level=args.cl, r_abs_accuracy=args.rAbsAcc,
                                 r_rel_accuracy=args.rRelAcc, strict_bounds=args.strictBounds))
        report.update(result, model=str(model_path),
            model_sha256={n: digest(model_path/n) for n in ('model.json','arrays.npz')})
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report.update(total_wall_seconds=time.perf_counter()-start,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for quantile, limit in zip(report['quantiles'], report['limits'], strict=True):
        print(f'Expected {100*quantile:4.1f}%: {poi} < {limit:.8g}')
