"""Capture an explicit subset of the unchanged 26-point native Asimov FD gate."""
import argparse
from collections import Counter
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from capture_asimov_native_reference import checked_native_value
from native_fd_scan import FRACTIONS, scan_coordinate
from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'generation-report', 'native-values', 'reference-report', 'directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.directory.exists()
    assert sorted(set(args.indices)) == args.indices and all(0 <= i < 26 for i in args.indices)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    generation = json.loads(args.generation_report.read_text())
    reference = json.loads(args.reference_report.read_text())
    assert generation['status'] == 'generated_and_identity_verified'
    assert generation['generation_kind'] == 'combine_prefit_asimov' and generation['MH'] == 125.
    assert sha(args.workspace) == generation['workspace_sha256'] == reference['workspace_sha256']
    assert reference['dataset'] == 'asimov_prefit' and reference['generation_report_sha256'] == sha(args.generation_report)
    assert sha(args.native_values) == reference['nll_arrays_sha256']
    with np.load(args.native_values, allow_pickle=False) as saved:
        names, points, expected = (saved[k] for k in ('parameter_names', 'points', 'values'))
    assert points.shape == (26, len(names)) and expected.shape == (26,) and np.isfinite(points).all()
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        fd_script_sha256=sha('scripts/native_fd_scan.py'),
        value_helper_sha256=sha('scripts/capture_asimov_native_reference.py'),
        dataset='asimov_prefit', workspace_sha256=sha(args.workspace),
        generation_report_sha256=sha(args.generation_report), native_values_sha256=sha(args.native_values),
        reference_report_sha256=sha(args.reference_report), selected_indices=args.indices,
        fractions=FRACTIONS, points=[],
        scope='Native-only subset of the previously fixed 26 Asimov points. Same five-point FD '
            'policy and step selection as the original scan; no new draws, AD arrays, fit or observed-data result.')
    args.directory.mkdir(parents=True)
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
            assert np.all(points >= lower) and np.all(points <= upper)
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('asimov_prefit'), config.GetNuisanceParameters())
            nll.setNoDeepLogEvalError(False)
            nll.setChannelMasks(config.GetPdf().channelMasks()); nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            value = lambda: checked_native_value(root, nll)
            for i in args.indices:
                for v, x in zip(variables, points[i], strict=True):
                    v.setVal(float(x))
                nll.clearZeroPoint()
                canonical = value()
                assert abs(canonical-expected[i]) < 1e-6
                nll.setZeroPoint(); zero = value()
                assert abs(zero) < 1e-6
                row = dict(index=i, canonical_nll=canonical, zero_point_nll=zero, cases=[])
                report.update(current_point=i, current_coordinate=0); save()
                for j, variable in enumerate(variables):
                    case = scan_coordinate(variable, value, scale=scales[j], lower=lower[j], upper=upper[j], zero=zero)
                    row['cases'].append(dict(index=j, name=str(names[j]), **case))
                    if (j+1) % 100 == 0:
                        report['current_coordinate'] = j+1; save()
                        print('point', i, 'coordinate', j+1, flush=True)
                nll.clearZeroPoint(); row['restoration_error'] = abs(value()-canonical)
                assert row['restoration_error'] < 1e-6
                row['status_counts'] = dict(Counter(c['status'] for c in row['cases']))
                artifact = args.directory/f'point_{i:02d}.json.gz'
                temporary = artifact.with_suffix('.tmp')
                temporary.write_bytes(gzip.compress(json.dumps(row, allow_nan=False).encode(), mtime=0))
                temporary.replace(artifact)
                report['points'].append(dict(index=i, artifact=str(artifact), artifact_sha256=sha(artifact),
                    status_counts=row['status_counts'], restoration_error=row['restoration_error']))
                save(); print('point complete', i, row['status_counts'], flush=True)
            nll.setAnalyticBarlowBeeston(False)
            del nll
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='shard_complete_pending_audit', root_only=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
