"""Native-v10 fixed-point diagnosis of the two saved Asimov fits; no minimization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    import numpy as np
    import ROOT as R
    from caching_nll_diagnostics import channel_nlls
    base = Path(os.environ['CMSSW_BASE'])
    library = base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
    config = json.loads((args.directory/'configuration.json').read_text())
    native = json.loads((args.directory/'results/gpu_hhvv_v10_a100_native_fit.json').read_text())
    driver = json.loads((args.directory/'results/gpu_hhvv_v10_a100_driver.json').read_text())
    workspace = args.directory/'generation/workspace_asimov.root'
    assert config['dataset'] == native['dataset'] == 'asimov_prefit'
    assert R.gROOT.GetVersion() == native['root_version'] == '6.30/07'
    assert sha(library) == native['library_sha256']
    assert sha(workspace) == native['workspace_sha256'] == config['workspace_sha256']
    R.gInterpreter.AddIncludePath(str(base/'src'))
    assert R.gSystem.Load(str(library)) >= 0
    assert R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    R.gROOT.SetBatch(True)
    with np.load(args.directory/'results/gpu_hhvv_v10_a100_fit.npz', allow_pickle=False) as saved:
        names = list(saved['parameter_names'])
        np.testing.assert_array_equal(names, native['fit']['parameters'])
        gpu = saved['fit_values'].copy()
        # Fit gradients are stored in free coordinates. free_gradient has
        # already removed the fit-coordinate scale; restore fixed entries as 0.
        free = np.array([float(lo) != float(hi) for lo, hi in zip(config['lower'], config['upper'])])
        assert int(free.sum()) == len(saved['fit_free_gradient'])
        gpu_gradient = np.zeros(len(names))
        gpu_gradient[free] = saved['fit_free_gradient']
    reference = np.asarray(native['fit']['values'])
    direction = gpu-reference
    handle = R.TFile.Open(str(workspace), 'READ')
    ws = handle.Get('w')
    mc = ws.obj('ModelConfig')
    variables = [ws.var(n) for n in names]
    for n in config['fit_state']['nuisance_bounds_removed']:
        ws.var(n).removeMin(); ws.var(n).removeMax()
    for flag, value in config['runtime_flags'].items():
        R.runtimedef.set(flag, value)
    nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), ws.data('asimov_prefit'), mc.GetNuisanceParameters())
    nll.setChannelMasks(mc.GetPdf().channelMasks())
    nll.setAnalyticBarlowBeeston(True)
    nll.clearConstantZeroPoint(); nll.clearZeroPoint()
    R.RooAbsReal.setEvalErrorLoggingMode(R.RooAbsReal.CountErrors)
    started = time.perf_counter()
    report = dict(status='running', script_sha256=sha(__file__), host=platform.node(),
        dataset='asimov_prefit', workspace_sha256=sha(workspace), library_sha256=sha(library),
        native_fit_sha256=sha(args.directory/'results/gpu_hhvv_v10_a100_native_fit.json'),
        gpu_fit_sha256=sha(args.directory/'results/gpu_hhvv_v10_a100_fit.npz'),
        root_version=R.gROOT.GetVersion(), scope='Fixed points only, no fit, no observed data. Native v10 segment, channel decomposition and finite differences; no acceptance thresholds changed.',
        segment=[], endpoint_derivatives=[], coordinates=[])
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    def value():
        R.RooAbsReal.clearEvalErrorLog()
        v = float(nll.getVal())
        assert np.isfinite(v) and R.RooAbsReal.numEvalErrors() == 0
        return v
    def point(x):
        for variable, coordinate in zip(variables, x):
            assert variable.getMin() <= coordinate <= variable.getMax()
            variable.setVal(float(coordinate))
        return value()
    def channels():
        terms, layout = channel_nlls(R, nll)
        report['channel_layout'] = layout
        return {str(t.pdf().GetName()): float(t.getVal()) for t in terms}
    endpoints = []
    for label, x, expected in [('native', reference, native['fit']['canonical_nll']),
                               ('gpu', gpu, native['tensor_point']['native_nll'])]:
        v = point(x)
        assert abs(v-expected) < 1e-6
        endpoints.append(dict(label=label, value=v, channels=channels()))
    a, b = endpoints
    rows = [dict(channel=n, native=a['channels'][n], gpu=b['channels'][n],
                 delta=b['channels'][n]-a['channels'][n]) for n in a['channels']]
    report['channel_deltas'] = sorted(rows, key=lambda r: abs(r['delta']), reverse=True)
    report['delta_nll'] = b['value']-a['value']
    report['nonchannel_delta'] = report['delta_nll']-sum(r['delta'] for r in rows)
    for alpha in [0., .01, .05, .075, .1, .125, .15, .25, .5, .75, .9, .99, 1.]:
        v = point(reference+alpha*direction)
        report['segment'].append(dict(alpha=alpha, nll=v, delta_from_native=v-a['value'],
            channel_deltas={n: y-a['channels'][n] for n, y in channels().items()}))
        save()
    # Per-channel zero-points suppress cancellation in the small endpoint probes.
    for label, x, sign in [('native', reference, 1.), ('gpu', gpu, -1.)]:
        nll.clearZeroPoint(); point(x); nll.setZeroPoint(); zero = value()
        probes = []
        for step in [.1, .03, .01, .003, .001, .0003]:
            v1 = point(x+sign*step*direction)-zero
            v2 = point(x+sign*2*step*direction)-zero
            probes.append(dict(step=step, first_delta=v1, second_delta=v2,
                directional_derivative=(4*v1-v2)/(2*sign*step),
                directional_curvature=(v2-2*v1)/(step*step)))
        report['endpoint_derivatives'].append(dict(endpoint=label, probes=probes,
            gpu_ad_directional_derivative=float(gpu_gradient@direction) if label=='gpu' else None))
        nll.clearZeroPoint(); save()
    selected = [r['name'] for r in driver['comparison']['top_parameters'][:3]]+['r_hhvv']
    for label, x in [('native', reference), ('gpu', gpu)]:
        for name in selected:
            i = names.index(name)
            nll.clearZeroPoint(); point(x); nll.setZeroPoint()
            probes = []
            scale = max(abs(native['fit']['errors'][i]), 1e-3)
            for fraction in [.05, .01, .002, .0004]:
                step = fraction*scale
                ys = []
                for multiple in [-2,-1,1,2]:
                    shifted = x.copy(); shifted[i] += multiple*step
                    ys.append(point(shifted))
                probes.append(dict(step=step, derivative=(8*(ys[2]-ys[1])-(ys[3]-ys[0]))/(12*step)))
            report['coordinates'].append(dict(endpoint=label, name=name, probes=probes,
                gpu_ad_gradient=float(gpu_gradient[i]) if label=='gpu' else None))
            nll.clearZeroPoint(); save()
    report['status'] = 'completed_fixed_point_diagnostic'
    nll.setAnalyticBarlowBeeston(False)
    save()
    print(report['status'], report['delta_nll'], report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
