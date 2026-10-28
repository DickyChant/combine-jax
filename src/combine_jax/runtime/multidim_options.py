"""Strict, explicit Asimov MultiDimFit command configuration."""
import math
from pathlib import Path

from .options import assignments, ranges


def add_multidim_command(commands):
    parser = commands.add_parser('multidim', help='Asimov MultiDimFit grids and batched profile impacts')
    parser.add_argument('--workspace', '-d', type=Path, required=True)
    parser.add_argument('--datacard', type=Path)
    parser.add_argument('--poi', '--pois', '--redefineSignalPOIs', dest='poi', required=True)
    parser.add_argument('--algo', choices=('grid', 'singles', 'impact', 'batch-impacts'), required=True)
    parser.add_argument('--parameters', default='')
    parser.add_argument('--mass', '-m', type=float, default=125.)
    parser.add_argument('--setParameters', default='')
    parser.add_argument('--setParameterRanges', default='')
    parser.add_argument('--freezeParameters', default='')
    parser.add_argument('--freezeNuisanceGroups', default='')
    parser.add_argument('--toys', '-t', type=int, choices=(-1,), required=True)
    parser.add_argument('--gridPoints', default='')
    parser.add_argument('--alignEdges', type=int, choices=(1,), default=1)
    parser.add_argument('--firstPoint', type=int, default=0)
    parser.add_argument('--lastPoint', type=int)
    parser.add_argument('--saveNLL', action='store_true')
    parser.add_argument('--robustFit', type=int, choices=(0, 1), default=0)
    parser.add_argument('--edm-validation', choices=('estimated', 'exact_hessian'), default='exact_hessian',
                        help='Exact final curvature checks EDM; both policies keep the 8 GiB guard')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--name', '-n', default='Test')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--discrete-policy', choices=('reject', 'fixed', 'profile'), default='reject')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--runtime', type=Path)


def request(args):
    from ..methods.multidim import grid_points
    pois = args.poi.split(',')
    values = assignments(args.setParameters)
    bounds = ranges(args.setParameterRanges)
    if not all(pois) or len(set(pois)) != len(pois):
        raise ValueError('Distinct nonempty POIs required')
    if set(pois)-values.keys():
        raise ValueError('Explicit nominal Asimov --setParameters values required for every POI')
    if not math.isfinite(args.mass) or '/' in args.name or '\\' in args.name:
        raise ValueError('Invalid mass or output name')
    if args.algo == 'grid':
        counts = [int(v) for v in args.gridPoints.split(',') if v]
        points = grid_points(pois, bounds, counts, args.firstPoint, args.lastPoint)
        parameters = []
    else:
        if len(pois) != 1:
            raise ValueError('Impacts and singles currently require one POI')
        parameters = args.parameters.split(',') if args.parameters else list(pois)
        if len(set(parameters)) != len(parameters) or not all(parameters):
            raise ValueError('Distinct impact parameters required')
        if args.algo == 'singles' and parameters != pois:
            raise ValueError('Singles supports the requested POI only')
        if args.algo == 'impact' and (len(parameters) != 1 or parameters == pois):
            raise ValueError('Impact requires one nuisance parameter')
        points = []
    return dict(method='MultiDimFit', algo=args.algo, pois=pois, poi=pois[0],
        workspace=str(args.workspace.resolve()), datacard=str(args.datacard.resolve()) if args.datacard else None,
        mass=args.mass, parameters=values, ranges=bounds, points=points, impact_parameters=parameters,
        freeze=list(filter(None, args.freezeParameters.split(','))),
        freeze_groups=list(filter(None, args.freezeNuisanceGroups.split(','))),
        run='expected', no_fit_asimov=True, signal_strength=values[pois[0]],
        backend=args.backend, discrete_policy=args.discrete_policy, template_convention='v10_density',
        float64=True, static_guard_gib=8., seed=args.seed, r_max=None,
        edm_validation=args.edm_validation, robust_fit=bool(args.robustFit))
