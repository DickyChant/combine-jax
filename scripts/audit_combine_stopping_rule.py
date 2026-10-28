"""Record installed Combine defaults and measure ROOT's NLL EDM threshold."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

from tensor_benchmark_common import save_json, sha


def probe():
    from combine_jax.root_io import load_caching_nll
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00'
    root.gInterpreter.Declare('double stopping_probe(double const* x) { return 0.5*(x[0]-2)*(x[0]-2); }')
    function = root.Math.Functor(root.stopping_probe, 1)
    minimizer = root.Math.Factory.CreateMinimizer('Minuit2', 'Migrad')
    minimizer.SetFunction(function)
    minimizer.SetVariable(0, 'x', 0., 1.)
    minimizer.SetErrorDef(.5)
    minimizer.SetTolerance(.1)
    minimizer.SetStrategy(1)
    minimizer.SetPrintLevel(2)
    assert minimizer.Minimize()
    x = root.RooRealVar('x', 'x', 0., -10., 10.)
    objective = root.RooFormulaVar('nll', '0.5*(@0-2)*(@0-2)', root.RooArgList(x))
    roo = root.RooMinimizer(objective)
    roo.setErrorLevel(.5)
    roo.setStrategy(1)
    roo.setEps(.1)
    roo.setPrintLevel(1)
    assert roo.minimize('Minuit2', 'Migrad') == 0
    assert 'jax' not in sys.modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        probe()
        return
    assert args.output and not args.output.exists()
    assert version('higgs-combine-tool') == '11.0.0'
    source = Path('vendor/combine_pypi/vendor/CombinedLimit/src/CascadeMinimizer.cc')
    header = source.parent.parent/'interface/CachingNLL.h'
    declarations = {}
    for key, expected in [('Type', '"Minuit2"'), ('Algo', '"Migrad"'),
                          ('Tolerance', '1e-1'), ('Precision', '-1.0')]:
        match = re.search(r'defaultMinimizer'+key+r'_\s*=\s*([^;]+);', source.read_text())
        assert match and match[1] == expected
        declarations[key] = match[0]
    assert re.search(r'int\s+CascadeMinimizer::strategy_=1;', source.read_text())
    assert 'Double_t defaultErrorLevel() const override { return 0.5; }' in header.read_text()
    logs = {}
    for label, command in [('cli', [str(Path(sys.prefix)/'bin/combine'), '-M', 'MultiDimFit', '--help']),
                           ('runtime', [sys.executable, __file__, '--probe'])]:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        path = args.output.with_name(args.output.stem+'_'+label+'.log')
        assert not path.exists()
        path.write_text(result.stdout)
        assert result.returncode == 0, path
        logs[label] = dict(path=str(path), sha256=sha(path), command=command)
    help_text = Path(logs['cli']['path']).read_text()
    defaults = {}
    for key in ('Type', 'Algo', 'Tolerance', 'Precision', 'Strategy'):
        match = re.search(r'--cminDefaultMinimizer'+key+r' arg \(=([^)]*)\)', help_text)
        assert match, key
        defaults[key] = match[1]
    assert defaults['Type'] == 'Minuit2' and defaults['Algo'] == 'Migrad'
    assert float(defaults['Tolerance']) == .1 and int(defaults['Strategy']) == 1
    thresholds = [float(v) for v in re.findall(r'Start iterating until Edm is < ([\deE.+-]+)',
                                              Path(logs['runtime']['path']).read_text())]
    assert thresholds == [.0001, .0001], thresholds
    save_json(args.output, dict(status='defaults_and_runtime_edm_verified',
        script_sha256=sha(__file__), combine_version=version('higgs-combine-tool'), root_version='6.40.00',
        sources={str(p): sha(p) for p in (source, header)}, source_declarations=declarations,
        cli_defaults=defaults, logs=logs, runtime_thresholds=thresholds,
        criterion=dict(tolerance=.1, strategy=1, error_definition=.5,
            normalized_edm_threshold=.002*.1, raw_nll_edm_threshold=.002*.1*.5,
            formula='EDM / Up < 0.002 * tolerance; Up=0.5 for Combine CachingSimNLL',
            interpretation='Use Combine own stopping rule. The user supplied 0.002*tolerance is the normalized EDM threshold; raw NLL EDM includes Up. Direct ROOT.Math and RooMinimizer runtime probes both print 0.0001.'),
        acceptance=dict(pending_question_resolved=True, strict_gradient='Separate diagnostic only; no new strict-gradient HHVV fits.',
            clipping='Four known MX2000 background clipping bins accepted without objective repair; covariance and uncertainties there are not certified.',
            comparison='Same-node, same-start, same-data native and tensor fits; compile separated, counts, canonical NLL difference, all-parameter max delta/sigma with r_hhvv and top nuisances explicit.',
            obsolete_native_benchmark='Earlier benchmark_combine_fit.py used strategy 2 and tolerance 1e-6; these attempts do not measure the new criterion.')))
    print('Combine 11: Minuit2/Migrad, strategy 1, tolerance 0.1; raw NLL EDM threshold 0.0001')


if __name__ == '__main__':
    main()
