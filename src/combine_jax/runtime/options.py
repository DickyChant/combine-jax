"""Strict Combine-style configuration for the expected-only public entrypoint."""
import math


def assignments(text):
    result = {}
    for item in filter(None, text.split(',')):
        name, value = item.split('=', 1)
        value = float(value)
        if not name or not math.isfinite(value):
            raise ValueError('Parameter assignments must have a name and finite value')
        # Combine/dhi can repeat a scanned parameter; the final assignment wins.
        result[name] = value
    return result


def ranges(text):
    result = {}
    for item in filter(None, text.split(':')):
        name, values = item.split('=', 1)
        low, high = map(float, values.split(','))
        if not name or not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Parameter ranges must be finite, increasing pairs')
        result[name] = [low, high]
    return result


def add_limits_command(commands):
    from pathlib import Path
    parser = commands.add_parser('limits', help='End-to-end blinded limits with standard Combine ROOT output')
    parser.add_argument('--workspace', '-d', type=Path, required=True)
    parser.add_argument('--datacard', type=Path, help='Optional additional channel-set check')
    parser.add_argument('--poi', '--pois', '--redefineSignalPOIs', dest='poi', required=True)
    parser.add_argument('--mass', '-m', type=float, default=125.)
    parser.add_argument('--setParameters', default='')
    parser.add_argument('--setParameterRanges', default='')
    parser.add_argument('--freezeParameters', default='')
    parser.add_argument('--freezeNuisanceGroups', default='')
    parser.add_argument('--run', choices=('blind','expected'), default='blind')
    parser.add_argument('--noFitAsimov', action='store_true')
    parser.add_argument('--rMax', type=float)
    parser.add_argument('--cl', type=float, default=.95)
    parser.add_argument('--rAbsAcc', type=float, default=.0005)
    parser.add_argument('--rRelAcc', type=float, default=.005)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--name', '-n', default='Test')
    parser.add_argument('--backend', choices=('cpu','gpu'), default='gpu')
    parser.add_argument('--discrete-policy', choices=('reject','fixed'), default='reject')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--runtime', type=Path, help='Runtime prefix created by scripts/setup_runtime.sh')


def request(args):
    if args.run == 'expected' and not args.noFitAsimov:
        raise ValueError('This end-to-end command is pre-fit only: use --run blind or --run expected --noFitAsimov')
    if ',' in args.poi or not args.poi:
        raise ValueError('AsymptoticLimits currently supports exactly one POI')
    if not math.isfinite(args.mass):
        raise ValueError('Mass must be finite')
    if '/' in args.name or '\\' in args.name:
        raise ValueError('Output name must not contain a path')
    if args.rMax is not None and (not math.isfinite(args.rMax) or args.rMax <= 0):
        raise ValueError('rMax must be positive and finite')
    return dict(workspace=str(args.workspace.resolve()),
        datacard=str(args.datacard.resolve()) if args.datacard else None,
        poi=args.poi, mass=args.mass, parameters=assignments(args.setParameters),
        ranges=ranges(args.setParameterRanges),
        freeze=[s for s in args.freezeParameters.split(',') if s],
        freeze_groups=[s for s in args.freezeNuisanceGroups.split(',') if s],
        r_max=args.rMax, run=args.run, no_fit_asimov=True, signal_strength=0.,
        backend=args.backend, discrete_policy=args.discrete_policy,
        template_convention='v10_density', float64=True, static_guard_gib=8.,
        seed=args.seed, confidence_level=args.cl, r_abs_accuracy=args.rAbsAcc,
        r_rel_accuracy=args.rRelAcc)
