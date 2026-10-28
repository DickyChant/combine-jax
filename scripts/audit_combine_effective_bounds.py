"""Probe Combine's installed bound policy and recover a saved native fit NLL."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys

from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    saved=json.loads(args.native_fit.read_text())
    assert saved['fit']['status']==0 and saved['fit']['edm_target_met']
    from combine_jax.root_io import load_caching_nll,open_workspace
    root=load_caching_nll()
    assert root.gROOT.GetVersion()=='6.40.00' and version('higgs-combine-tool')=='11.0.0'
    assert root.gInterpreter.Declare(r'''
        #include "HiggsAnalysis/CombinedLimit/interface/MultiDimFit.h"
        namespace combine_jax_bounds_audit {
        class Probe : public MultiDimFit {
        public:
            using FitterAlgoBase::optimizeBounds;
            using FitterAlgoBase::restoreBounds;
        };
        }
    ''')
    root.runtimedef.set('OPTIMIZE_BOUNDS',1)
    root.runtimedef.set('UNBOUND_GAUSSIANS',0)
    probe=root.combine_jax_bounds_audit.Probe()
    names=saved['fit']['parameters']
    paths=['vendor/combine_pypi/vendor/CombinedLimit/bin/combine.cpp',
           'vendor/combine_pypi/vendor/CombinedLimit/src/FitterAlgoBase.cc',
           'vendor/combine_pypi/vendor/CombinedLimit/src/MultiDimFit.cc']
    report=dict(status='checking',script_sha256=sha(__file__),source_sha256={p:sha(p) for p in paths},
        native_fit=str(args.native_fit),native_fit_sha256=sha(args.native_fit),
        root_version=root.gROOT.GetVersion(),combine_version=version('higgs-combine-tool'),
        policy=dict(OPTIMIZE_BOUNDS=1,UNBOUND_GAUSSIANS=0,
            rule='FitterAlgoBase::optimizeBounds removes both bounds from ModelConfig nuisance RooRealVars marked optimizeBounds; restoreBounds runs after runSpecific. No histogram or likelihood repair.'),
        parameters=[],evaluations=[])
    workspace_path=Path(saved['workdir'])/saved['prepared_workspace']
    assert sha(workspace_path)==saved['prepared_workspace_sha256']
    def evaluate(workspace,label):
        config=workspace.obj('ModelConfig')
        nll=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('data_obs'),config.GetNuisanceParameters())
        if config.GetPdf().ClassName()=='RooSimultaneousOpt':nll.setChannelMasks(config.GetPdf().channelMasks())
        nll.setAnalyticBarlowBeeston(True);nll.clearConstantZeroPoint();nll.clearZeroPoint()
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors);root.RooAbsReal.clearEvalErrorLog()
        result=dict(label=label,value=float(nll.getVal()),roofit_errors=int(root.RooAbsReal.numEvalErrors()))
        nll.setAnalyticBarlowBeeston(False);del nll
        report['evaluations'].append(result);save_json(args.output,report)
        return result['value']
    with open_workspace(workspace_path) as workspace:
        config=workspace.obj('ModelConfig');nuisances=config.GetNuisanceParameters()
        for name,initial,value in zip(names,saved['initial_values'],saved['fit']['values'],strict=True):
            var=workspace.var(name);assert var.getVal()==initial
            report['parameters'].append(dict(name=name,initial=initial,fitted=value,
                constant=bool(var.isConstant()),nuisance=bool(nuisances.find(name)),
                optimize_bounds=bool(var.getAttribute('optimizeBounds')),
                stored_lower=float(var.getMin()),stored_upper=float(var.getMax()),
                stored_has_min=bool(var.hasMin()),stored_has_max=bool(var.hasMax())))
        original=evaluate(workspace,'nominal_stored_bounds')
        probe.optimizeBounds(workspace,config)
        for row in report['parameters']:
            var=workspace.var(row['name'])
            row.update(effective_lower=float(var.getMin()),effective_upper=float(var.getMax()),
                effective_has_min=bool(var.hasMin()),effective_has_max=bool(var.hasMax()))
            expected=row['nuisance'] and row['optimize_bounds']
            if expected:assert not var.hasMin() and not var.hasMax(),row['name']
            else:assert var.getMin()==row['stored_lower'] and var.getMax()==row['stored_upper'],row['name']
            row['changed']=var.getMin()!=row['stored_lower'] or var.getMax()!=row['stored_upper']
            row['fit_outside_stored_bounds']=not row['stored_lower']<=row['fitted']<=row['stored_upper']
            assert var.getMin()<=row['fitted']<=var.getMax(),row['name']
        effective=evaluate(workspace,'nominal_effective_bounds')
        for row in report['parameters']:workspace.var(row['name']).setVal(row['fitted'])
        fitted=evaluate(workspace,'native_fit_effective_bounds')
        for row in report['parameters']:workspace.var(row['name']).setVal(row['initial'])
        restored=evaluate(workspace,'restored_nominal_effective_bounds')
        probe.restoreBounds(workspace,config)
        assert all(workspace.var(r['name']).getMin()==r['stored_lower'] and
                   workspace.var(r['name']).getMax()==r['stored_upper'] for r in report['parameters'])
    assert not any(r['roofit_errors'] for r in report['evaluations'])
    assert abs(original-effective)<1e-6 and abs(restored-effective)<1e-6
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    report.update(status='passed_effective_bounds_probe',changed_parameters=sum(r['changed'] for r in report['parameters']),
        outside_stored_bounds=[r['name'] for r in report['parameters'] if r['fit_outside_stored_bounds']],
        nominal_nll_bound_policy_difference=effective-original,native_canonical_nll=fitted,
        nominal_restoration_error=abs(restored-effective),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='Exact installed native bound-policy call, without minimization; source/CLI default policy is explicit. Recovers the native canonical NLL under its actual fit domain. Existing tensor fit metadata still uses stored bounds, so no same-domain speed claim follows.')
    save_json(args.output,report)
    print(report['status'],report['changed_parameters'],report['outside_stored_bounds'],fitted,flush=True)


if __name__=='__main__':main()
