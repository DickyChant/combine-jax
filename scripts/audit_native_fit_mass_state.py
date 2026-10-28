"""Recover a CLI fit's NLL with its saved fixed state, without minimization."""
import argparse
import gzip
import json
from pathlib import Path
import re

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-audit', type=Path, required=True)
    parser.add_argument('--native-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    audit = json.loads(args.workspace_audit.read_text())
    native = json.loads(Path(audit['native_fit']).read_text())
    assert sha(audit['native_fit']) == audit['native_fit_sha256']
    record = audit['workspaces'][1]
    assert sha(record['path']) == record['sha256']
    text = gzip.open(args.native_log, 'rt').read()
    values = re.findall(r'Minuit2Minimizer : Valid minimum - status = 0\s+FVAL\s*=\s*([\deE.+-]+)', text)
    assert len(values) == 1
    from combine_jax.root_io import load_caching_nll, open_workspace
    root = load_caching_nll()
    result = dict(status='checking', script_sha256=sha(__file__),
        workspace_audit=str(args.workspace_audit), workspace_audit_sha256=sha(args.workspace_audit),
        native_log=str(args.native_log), native_log_sha256=sha(args.native_log),
        root_version=root.gROOT.GetVersion(), native_internal_nll=float(values[0]), evaluations=[])
    save_json(args.output, result)
    for snapshot in ('clean', 'MultiDimFit'):
        with open_workspace(record['path']) as workspace:
            config = workspace.obj('ModelConfig')
            # Snapshot MultiDimFit contains the effective fit ranges; clean contains
            # the original ranges. Remove the marked nuisance bounds for both.
            assert workspace.getSnapshot(snapshot)
            workspace.loadSnapshot(snapshot)
            for var in config.GetNuisanceParameters():
                if var.InheritsFrom('RooRealVar') and var.getAttribute('optimizeBounds'):
                    var.removeMin(); var.removeMax()
            expected = native['initial_values'] if snapshot == 'clean' else native['fit']['values']
            assert [workspace.var(n).getVal() for n in native['initial_parameters']] == expected
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
            root.RooAbsReal.clearEvalErrorLog()
            result['evaluations'].append(dict(snapshot=snapshot, mh=float(workspace.var('MH').getVal()),
                canonical_nll=float(nll.getVal()), roofit_errors=int(root.RooAbsReal.numEvalErrors())))
            nll.setAnalyticBarlowBeeston(False)
            del nll
            save_json(args.output, result)
    initial, fitted = result['evaluations']
    result['offset_error'] = abs(fitted['canonical_nll']-initial['canonical_nll']-result['native_internal_nll'])
    assert all(r['roofit_errors'] == 0 for r in result['evaluations'])
    assert result['offset_error'] < 1e-6
    result.update(status='passed_saved_state_recovery',
        scope='Native CLI ran at MH=120; tensor model was imported at MH=125. This recovers the actual native objective and verifies its zero-point relation. The MH=125 evaluation in the earlier effective-bounds probe is a cross-evaluation, not the native fit NLL. No paired fit acceptance or speed follows.')
    save_json(args.output, result)
    print(result['status'], result['evaluations'], result['offset_error'], flush=True)


if __name__ == '__main__':
    main()
