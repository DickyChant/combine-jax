"""Fresh native NLL and AD-independent FD references for the nominal HHVV Asimov."""
import argparse
from collections import Counter
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
from native_fd_scan import FRACTIONS, scan_coordinate
from tensor_benchmark_common import Resources, provenance, save_json, sha


def checked_native_value(root,nll):
    root.RooAbsReal.clearEvalErrorLog();result=float(nll.getVal())
    errors=int(root.RooAbsReal.numEvalErrors())
    if not np.isfinite(result) or errors or result>=9e9:
        raise ValueError(f'Invalid native NLL or FASTEXIT sentinel; RooFit errors={errors}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','generation-report','points','directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--nll-only',action='store_true')
    parser.add_argument('--generate-points',action='store_true',help='Draw and retain 25 native-valid Asimov random points')
    parser.add_argument('--seed',type=int,default=20007319)
    args=parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))])
    generation=json.loads(args.generation_report.read_text())
    assert generation['status']=='generated_and_identity_verified'
    assert generation['generation_kind']=='combine_prefit_asimov' and generation['MH']==125.
    assert sha(args.workspace)==generation['workspace_sha256']
    from combine_jax.root_io import load_caching_nll,open_workspace,REFERENCE_FLAGS
    from combine_jax.dataset_identity import workspace_dataset_identity
    from combine_jax.tensor.fit_policy import combine_v11_fit_context
    root=load_caching_nll()
    assert root.gROOT.GetVersion()=='6.40.00' and version('higgs-combine-tool')=='11.0.0'
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    with np.load(args.points,allow_pickle=False) as saved:
        names=saved['parameter_names'];points=saved['points']
    assert points.shape==(26,len(names)) and np.isfinite(points).all()
    report=dict(provenance(),status='running',script_sha256=sha(__file__),
        fd_script_sha256=sha('scripts/native_fd_scan.py'),dataset='asimov_prefit',
        workspace_sha256=sha(args.workspace),generation_report_sha256=sha(args.generation_report),
        points_sha256=sha(args.points),root_version=root.gROOT.GetVersion(),combine_version=version('higgs-combine-tool'),
        runtime_flags=REFERENCE_FLAGS,fractions=FRACTIONS,points=[],rejected_points=[],
        native_domain_check=dict(no_deep_log_eval_error=False,reject_nll_at_or_above=9e9,
            require_zero_roofit_errors=True),
        sampling=dict(generate_points=args.generate_points,seed=args.seed,
            rule='Uniform +/-2 nominal errors, fallback one; clip within effective native bounds with 1% margin; fixed coordinates held. Reject native RooFit errors or FASTEXIT sentinel, at most 250 draws. Native selection only, no AD values.'),
        selection='Unchanged five-point native FD policy: first three consecutive valid estimates with adjacent relative differences <1e-4; derivatives smaller than 1e-4 remain roundoff-sensitive. No AD arrays are read. Four fixed coordinates are excluded.',
        scope='Fresh Asimov native reference generation, no fit. Unresolved references remain explicit and cannot establish gradient acceptance.')
    args.directory.mkdir(parents=True);save_json(args.output,report)
    monitor=Resources(False,args.output);started=time.perf_counter()
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,resources=monitor.snapshot())
        save_json(args.output,report)
    try:
        with open_workspace(args.workspace) as workspace,combine_v11_fit_context(workspace):
            assert workspace.var('MH').getVal()==125. and workspace.var('MX').getVal()==2000.
            assert workspace_dataset_identity(workspace,'asimov_prefit')==generation['dataset']['identity']
            variables=[workspace.var(str(n)) for n in names]
            np.testing.assert_array_equal([v.getVal() for v in variables],points[0])
            lower=np.array([v.getVal() if v.isConstant() else v.getMin() for v in variables])
            upper=np.array([v.getVal() if v.isConstant() else v.getMax() for v in variables])
            scales=np.array([v.getError() if v.getError()>0 else max(abs(v.getVal()),1.) for v in variables])
            assert np.all(points>=lower) and np.all(points<=upper)
            config=workspace.obj('ModelConfig')
            nll=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('asimov_prefit'),config.GetNuisanceParameters())
            nll.setNoDeepLogEvalError(False)
            nll.setChannelMasks(config.GetPdf().channelMasks());nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint();nll.clearZeroPoint()
            def value():
                return checked_native_value(root,nll)
            def assign(point):
                for v,x in zip(variables,point,strict=True):v.setVal(float(x))
            values=[]
            if args.generate_points:
                initial=points[0].copy();chosen=[];rng=np.random.default_rng(args.seed)
                draw_scales=np.array([v.getError() if v.getError()>0 else 1. for v in variables])
                margin=np.minimum(.01*draw_scales,.01*(upper-lower))
                for attempt in range(251):
                    point=initial.copy() if attempt==0 else np.clip(initial+rng.uniform(-2,2,len(names))*draw_scales,
                        lower+margin,upper-margin)
                    assign(point)
                    try:native_value=value()
                    except ValueError as exc:
                        if attempt==0:raise
                        report['rejected_points'].append(dict(attempt=attempt,point=point.tolist(),reason=str(exc)))
                        save();print('rejected draw',attempt,flush=True);continue
                    chosen.append(point);values.append(native_value)
                    report['accepted_draws']=report.get('accepted_draws',[])+[attempt]
                    save();print('accepted point',len(chosen)-1,'draw',attempt,flush=True)
                    if len(chosen)==26:break
                assert len(chosen)==26,'Fewer than 25 native-valid random points after 250 draws'
                points=np.asarray(chosen)
            else:
                for i,point in enumerate(points):
                    assign(point);values.append(value())
                    report['nll_points']=i+1;save();print('native NLL point',i,flush=True)
            nll_path=args.directory/'values.npz'
            np.savez_compressed(nll_path,parameter_names=names,points=points,values=np.array(values))
            report.update(nll_arrays=str(nll_path),nll_arrays_sha256=sha(nll_path),nll_points=len(values))
            save();print('native Asimov NLL points captured',len(values),flush=True)
            if not args.nll_only:
                for i,point in enumerate(points):
                    assign(point);nll.clearZeroPoint();canonical=value()
                    assert abs(canonical-values[i])<1e-6
                    nll.setZeroPoint();zero=value();assert abs(zero)<1e-6
                    row=dict(index=i,canonical_nll=canonical,zero_point_nll=zero,cases=[])
                    report['current_point']=i;save()
                    for j,v in enumerate(variables):
                        case=scan_coordinate(v,value,scale=scales[j],lower=lower[j],upper=upper[j],zero=zero)
                        row['cases'].append(dict(index=j,name=str(names[j]),**case))
                        if (j+1)%200==0:
                            report['current_coordinate']=j+1;save()
                            print('point',i,'coordinate',j+1,'elapsed',time.perf_counter()-started,flush=True)
                    nll.clearZeroPoint();row['restoration_error']=abs(value()-canonical)
                    assert row['restoration_error']<1e-6
                    row['status_counts']=dict(Counter(c['status'] for c in row['cases']))
                    artifact=args.directory/f'point_{i:02d}.json.gz'
                    artifact.write_bytes(gzip.compress(json.dumps(row,allow_nan=False).encode(),mtime=0))
                    report['points'].append(dict(index=i,artifact=str(artifact),artifact_sha256=sha(artifact),
                        status_counts=row['status_counts'],restoration_error=row['restoration_error']))
                    save();print('point complete',i,row['status_counts'],flush=True)
            nll.setAnalyticBarlowBeeston(False);del nll
        ready=all(set(p['status_counts'])<={'converged','fixed_bounds'} for p in report['points'])
        report['status']='nll_captured' if args.nll_only else ('reference_ready' if ready else 'reference_uncertain')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        monitor.close();save()


if __name__=='__main__':main()
