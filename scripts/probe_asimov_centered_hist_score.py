"""Native-only Asimov histogram-score FD diagnostic with exact normalization centering."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from capture_asimov_native_reference import checked_native_value
from native_cached_deltas import CachedDeltas
from native_centered_hist_score import CenteredHistogramScore
from native_fd_refinement import REFINEMENT_FRACTIONS, refine_coordinate
from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'generation-report', 'native-values', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [cpus[min(1, len(cpus)-1)]])
    generation = json.loads(args.generation_report.read_text())
    assert generation['status'] == 'generated_and_identity_verified'
    assert generation['generation_kind'] == 'combine_prefit_asimov' and generation['MH'] == 125.
    assert sha(args.workspace) == generation['workspace_sha256']
    selected = json.loads(args.selection.read_text())
    assert selected and all(set(c) == {'point', 'index', 'name'} for c in selected)
    assert len({(c['point'], c['index']) for c in selected}) == len(selected)
    with np.load(args.native_values, allow_pickle=False) as archive:
        names, points, expected = (archive[k] for k in ('parameter_names', 'points', 'values'))
    assert points.shape == (26, len(names)) and expected.shape == (26,)
    for c in selected:
        assert 0 <= c['point'] < 26 and 0 <= c['index'] < len(names) and names[c['index']] == c['name']
    report = dict(provenance(), status='running', dataset='asimov_prefit',
        workspace_sha256=sha(args.workspace), generation_report_sha256=sha(args.generation_report),
        native_values_sha256=sha(args.native_values), selection_sha256=sha(args.selection),
        artifacts={p: sha(p) for p in (__file__, 'scripts/native_cached_deltas.py',
            'scripts/native_fd_refinement.py', 'scripts/native_fd_scan.py', 'scripts/native_centered_hist_score.py')},
        reference_method='native_centered_histogram_score',
        fractions=REFINEMENT_FRACTIONS, longdouble_mantissa_bits=np.finfo(np.longdouble).nmant, cases=[],
        scope='Native-only diagnostic; no fitted or observed-data result, JAX, or Torch is read. '
            'Finite differences of the first-order NLL linearization use native component caches and exact full-grid histogram normalization. '
            'Every NLL difference must match live Combine within 1e-6. '
            'Unsupported dependence remains explicit. Reference selection uses the fixed 19-step grid without AD; this diagnostic alone does not establish acceptance.')
    save_json(args.output, report)
    monitor = Resources(False, args.output, rss_limit_mib=6144)
    started = time.perf_counter()
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started, resources=monitor.snapshot())
        save_json(args.output, report)
    try:
        from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
        from combine_jax.dataset_identity import workspace_dataset_identity
        from combine_jax.tensor.fit_policy import combine_v11_fit_context
        root = load_caching_nll()
        assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
        report.update(root_version=root.gROOT.GetVersion(), combine_version=version('higgs-combine-tool'), runtime_flags=REFERENCE_FLAGS)
        with open_workspace(args.workspace) as workspace, combine_v11_fit_context(workspace):
            assert workspace_dataset_identity(workspace, 'asimov_prefit') == generation['dataset']['identity']
            variables = [workspace.var(str(n)) for n in names]
            np.testing.assert_array_equal([v.getVal() for v in variables], points[0])
            lower = [v.getVal() if v.isConstant() else v.getMin() for v in variables]
            upper = [v.getVal() if v.isConstant() else v.getMax() for v in variables]
            scales = [v.getError() if v.getError() > 0 else max(abs(v.getVal()), 1.) for v in variables]
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('asimov_prefit'), config.GetNuisanceParameters())
            nll.setNoDeepLogEvalError(False)
            nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            value = lambda: checked_native_value(root, nll)
            value()
            stable = CachedDeltas(root, nll)
            score = CenteredHistogramScore(stable)
            for point in sorted({c['point'] for c in selected}):
                for v, x in zip(variables, points[point], strict=True):
                    v.setVal(float(x))
                nll.clearZeroPoint()
                canonical = value()
                assert abs(canonical-expected[point]) < 1e-6
                nll.setZeroPoint()
                assert abs(value()) < 1e-6
                for case in (c for c in selected if c['point'] == point):
                    j = case['index']
                    report['current_case'] = case; save()
                    try:
                        score.reset(variables[j])
                    except NotImplementedError as exc:
                        report['cases'].append(dict(**case, canonical_nll=canonical,
                            canonical_reference_error=abs(canonical-expected[point]),
                            reference=dict(status='unsupported',coordinate=float(variables[j].getVal()),probes=[]),
                            reason=str(exc)))
                        save(); continue
                    assert score.value() == 0.
                    result = refine_coordinate(variables[j], score.value, scale=scales[j],
                        lower=lower[j], upper=upper[j], zero=0.)
                    report['cases'].append(dict(**case, canonical_nll=canonical,
                        canonical_reference_error=abs(canonical-expected[point]), reference=result,
                        channels=[str(c.channel.pdf().GetName()) for c in stable.selected],
                        normalized_components=score.evidence,
                        max_delta_parity_error=stable.max_delta_parity_error))
                    save(); print('probed', point, case['name'], result['status'], flush=True)
                nll.clearZeroPoint()
                assert abs(value()-canonical) < 1e-6
            report.update(status='diagnostic_complete_pending_audit', root_only=True,
                max_delta_parity_error=stable.max_delta_parity_error, evaluations=stable.evaluations,
                reflected_members=list(stable.reader.layout.values()),
                status_counts=dict(Counter(c['reference']['status'] for c in report['cases'])))
            nll.setAnalyticBarlowBeeston(False)
            del score, stable, nll
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
