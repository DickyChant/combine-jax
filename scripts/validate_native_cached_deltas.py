"""Check stable native-cache deltas against Combine on the generic cards."""
import argparse
import json
from pathlib import Path

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.fit_policy import combine_v11_fit_context
from native_cached_deltas import CachedDeltas
from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest = json.loads(args.manifest.read_text())
    root = load_caching_nll()
    report = dict(status='checking', root_version=root.gROOT.GetVersion(), cases=[],
        artifacts={str(p): sha(p) for p in (args.manifest, Path(__file__), Path('scripts/native_cached_deltas.py'))})
    for case in ('counting', 'automcstats', 'parametric', 'multipdf_exp', 'multipdf_bern'):
        workspace_path = manifest[case]['workspace']
        with open_workspace(workspace_path) as workspace, combine_v11_fit_context(workspace):
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            nll.getVal()
            stable = CachedDeltas(root, nll)
            checked = []
            with np.load(manifest[case]['points'], allow_pickle=False) as saved:
                names = saved['parameter_names']
            for name in names:
                variable = workspace.var(str(name))
                if variable.isConstant() or variable.getMin() == variable.getMax():
                    continue
                stable.reset(variable)
                original = float(variable.getVal())
                scale = variable.getError() if variable.getError() > 0 else max(1., abs(original))
                for sign in (-1, 1):
                    shifted = original+sign*.01*scale
                    if not variable.getMin() <= shifted <= variable.getMax():
                        continue
                    try:
                        variable.setVal(shifted)
                        delta = stable.value()
                        checked.append(dict(name=str(name), sign=sign, delta=delta))
                    finally:
                        variable.setVal(original)
                        assert abs(stable.value()) < 1e-14
            report['cases'].append(dict(case=case, workspace_sha256=sha(workspace_path),
                max_delta_parity_error=stable.max_delta_parity_error, probes=checked))
            assert stable.max_delta_parity_error < 1e-10
            nll.setAnalyticBarlowBeeston(False)
            del stable, nll
    report['status'] = 'passed'
    save_json(args.output, report)
    print('passed', len(report['cases']), 'cards, max error', max(c['max_delta_parity_error'] for c in report['cases']))


if __name__ == '__main__':
    main()
