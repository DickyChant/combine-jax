"""Native-only Asimov FD refinement at explicitly selected point/parameter pairs."""
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
from native_fd_refinement import REFINEMENT_FRACTIONS,REFERENCE_SPREAD,refine_coordinate
from tensor_benchmark_common import Resources,provenance,save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','generation-report','native-values','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[min(1,len(cpus)-1)]])
    generation=json.loads(args.generation_report.read_text())
    assert generation['status']=='generated_and_identity_verified' and generation['generation_kind']=='combine_prefit_asimov'
    assert generation['MH']==125. and sha(args.workspace)==generation['workspace_sha256']
    selected=json.loads(args.selection.read_text())
    assert selected and all(set(c)=={'point','index','name'} for c in selected)
    assert len({(c['point'],c['index']) for c in selected})==len(selected)
    with np.load(args.native_values,allow_pickle=False) as archive:
        names=archive['parameter_names'];points=archive['points'];expected=archive['values']
    assert points.shape==(26,len(names)) and expected.shape==(26,)
    for case in selected:
        assert 0<=case['point']<26 and 0<=case['index']<len(names) and names[case['index']]==case['name']
    report=dict(provenance(),status='running',dataset='asimov_prefit',script_sha256=sha(__file__),
        helper_sha256=sha('scripts/native_fd_refinement.py'),base_fd_sha256=sha('scripts/native_fd_scan.py'),
        workspace_sha256=sha(args.workspace),generation_report_sha256=sha(args.generation_report),
        native_values_sha256=sha(args.native_values),selection_sha256=sha(args.selection),
        fractions=REFINEMENT_FRACTIONS,reference_spread=REFERENCE_SPREAD,cases=[],
        policy='Capture the full fixed native step grid. Select the first triple of valid, distinct decreasing steps with identical stencil/direction and full relative spread <1e-5. Derivatives below 1e-4 remain uncertified. Later AD acceptance remains relative <1e-4 against all three references.',
        scope='Native reference generation only; the selection file contains point/parameter identifiers and no AD values. No fit is performed; no observed-data result is read.')
    save_json(args.output,report);monitor=Resources(False,args.output,rss_limit_mib=6144);started=time.perf_counter()
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,resources=monitor.snapshot());save_json(args.output,report)
    try:
        from combine_jax.root_io import load_caching_nll,open_workspace,REFERENCE_FLAGS
        from combine_jax.dataset_identity import workspace_dataset_identity
        from combine_jax.tensor.fit_policy import combine_v11_fit_context
        root=load_caching_nll();assert root.gROOT.GetVersion()=='6.40.00' and version('higgs-combine-tool')=='11.0.0'
        report.update(root_version=root.gROOT.GetVersion(),combine_version=version('higgs-combine-tool'),runtime_flags=REFERENCE_FLAGS)
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
        with open_workspace(args.workspace) as workspace,combine_v11_fit_context(workspace):
            assert workspace_dataset_identity(workspace,'asimov_prefit')==generation['dataset']['identity']
            variables=[workspace.var(str(n)) for n in names]
            np.testing.assert_array_equal([v.getVal() for v in variables],points[0])
            lower=[v.getVal() if v.isConstant() else v.getMin() for v in variables]
            upper=[v.getVal() if v.isConstant() else v.getMax() for v in variables]
            scales=[v.getError() if v.getError()>0 else max(abs(v.getVal()),1.) for v in variables]
            config=workspace.obj('ModelConfig')
            nll=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('asimov_prefit'),config.GetNuisanceParameters())
            nll.setNoDeepLogEvalError(False);nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True);nll.clearConstantZeroPoint();nll.clearZeroPoint()
            value=lambda:checked_native_value(root,nll)
            for point in sorted({c['point'] for c in selected}):
                for v,x in zip(variables,points[point],strict=True):v.setVal(float(x))
                nll.clearZeroPoint();canonical=value();assert abs(canonical-expected[point])<1e-6
                nll.setZeroPoint();zero=value();assert abs(zero)<1e-6
                for case in (c for c in selected if c['point']==point):
                    j=case['index'];report['current_case']=case;save()
                    result=refine_coordinate(variables[j],value,scale=scales[j],lower=lower[j],upper=upper[j],zero=zero)
                    row=dict(**case,canonical_nll=canonical,canonical_reference_error=abs(canonical-expected[point]),reference=result)
                    report['cases'].append(row);save();print('refined',point,case['name'],result['status'],flush=True)
                nll.clearZeroPoint();assert abs(value()-canonical)<1e-6
            nll.setAnalyticBarlowBeeston(False);del nll
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        counts=Counter(c['reference']['status'] for c in report['cases'])
        report.update(status='refinement_complete_pending_audit',status_counts=dict(counts),root_only=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        monitor.close();save()


if __name__=='__main__':main()
