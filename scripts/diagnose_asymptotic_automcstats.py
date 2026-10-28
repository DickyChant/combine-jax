"""Cross Asimov preparation and profile engines on the synthetic autoMCStats card."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import numpy as np

from tensor_benchmark_common import sha, save_json
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.dataset_identity import workspace_dataset_identity
from combine_jax.tensor.model import from_workspace, make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.methods.runner import run_expected_model


def weights(data):
    result = []
    for i in range(data.numEntries()):
        data.get(i)
        result.append(float(data.weight()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jax-report',type=Path,required=True)
    args = parser.parse_args()
    directory,output = args.directory,args.output
    assert not directory.exists() and not output.exists()
    directory.mkdir()
    source = Path('build/tensor_native_policy_inputs/automcstats/workspace.root')
    card = Path('build/tensor_native_policy_inputs/automcstats/card.txt')
    previous = json.loads(args.jax_report.read_text())
    root = load_caching_nll()
    root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"')
    root.Math.MinimizerOptions.SetDefaultMinimizer('Minuit2','Migrad')
    root.Math.MinimizerOptions.SetDefaultStrategy(1)
    root.Math.MinimizerOptions.SetDefaultTolerance(.1)
    report = dict(status='running', script_sha256=sha(__file__), source_workspace_sha256=sha(source),
        jax_report=str(args.jax_report),jax_report_sha256=sha(args.jax_report),
        method_sources={str(p):sha(p) for p in Path('src/combine_jax/methods').glob('*.py')},
        scope='Synthetic autoMCStats only; cross native and JAX background-Asimov generation with native/JAX expected profiles. No HHVV fits.', cases=[])
    save_json(output,report)
    prepared = directory/'native_prepared.root'
    with open_workspace(source) as ws:
        config = ws.obj('ModelConfig')
        snapshot = root.RooArgSet()
        generated = root.asimovutils.asimovDatasetWithFit(config,ws.data('data_obs'),snapshot,True,0.,-1)
        report['native_background_parameters'] = {str(v.GetName()):float(v.getVal()) for v in config.GetNuisanceParameters()}
        report['native_globals'] = {str(v.GetName()):float(v.getVal()) for v in snapshot}
        for v in snapshot:
            ws.var(v.GetName()).setVal(v.getVal())
        report['native_asimov_weights'] = weights(generated)
        getattr(ws,'import')(generated,root.RooFit.Rename('combine_jax_asimov_bonly'))
        ws.var('r').setConstant(False)
        ws.var('r').setMin(0.)
        ws.var('r').setVal(0.)
        ws.writeToFile(str(prepared))
        model = from_workspace(ws,card,data='combine_jax_asimov_bonly')
        identity = workspace_dataset_identity(ws,'combine_jax_asimov_bonly')
        model.metadata['asymptotic_asimov'] = dict(poi='r',signal_strength=0.,mode='conditional_background_fit',
            dataset='combine_jax_asimov_bonly',dataset_identity=identity)
        model.save(directory/'native_model')
        result = run_expected_model(model,'r')
        report['jax_profiles_native_asimov'] = result
    with open_workspace(previous['preparation']['prepared_workspace']) as ws:
        report['jax_background_parameters'] = {str(v.GetName()):float(v.getVal()) for v in ws.obj('ModelConfig').GetNuisanceParameters()}
        report['jax_globals'] = previous['dataset']['globals']
        report['jax_asimov_weights'] = weights(ws.data('combine_jax_asimov_bonly'))
    report['jax_profiles_jax_asimov'] = dict(limits=previous['limits'],minimum_nll=previous['minimum_nll'],
        conditional=previous['preparation']['conditional_fit'],profile_fits=previous['profile_fits'])
    save_json(output,report)
    for label,workspace in [('native',prepared),('jax',Path(previous['preparation']['prepared_workspace']))]:
        work = directory/label
        work.mkdir()
        command = ['combine','-M','AsymptoticLimits','-d',str(workspace.resolve()),'--run','blind','-m','125','-v','0']
        start = time.perf_counter()
        with (work/'combine.log').open('x') as log:
            run = subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        assert run.returncode == 0
        file = next(work.glob('higgsCombine*.root'))
        handle = root.TFile.Open(str(file))
        limits = sorted((float(row.quantileExpected),float(row.limit)) for row in handle.Get('limit') if row.quantileExpected>0)
        handle.Close()
        assert len(limits)==5
        report['cases'].append(dict(preparation=label,engine='native',quantiles=[p for p,x in limits],
            limits=[x for p,x in limits],command=command,wall_seconds=time.perf_counter()-start,
            workspace_sha256=sha(workspace),output_sha256=sha(file)))
        save_json(output,report)
    # Same-point postfit-Asimov NLL and finite-difference gates, independently of
    # optimizer convergence. These catch changed global/BB constraint semantics.
    with open_workspace(prepared) as ws:
        config = ws.obj('ModelConfig')
        nll = root.cacheutils.CachingSimNLL(config.GetPdf(),ws.data('combine_jax_asimov_bonly'),config.GetNuisanceParameters())
        nll.setAnalyticBarlowBeeston(True);nll.clearConstantZeroPoint();nll.clearZeroPoint()
        info = model_parameter_info(model)
        names = info.metadata['names']
        variables = [ws.var(n) for n in names]
        function = jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
        arrays = jax.device_put(dict(model.arrays))
        point = info.arrays['initial'].copy()
        point[names.index('r')] = .9
        def native(x):
            for v,a in zip(variables,x,strict=True):v.setVal(float(a))
            return float(nll.getVal())
        value,grad = function(point,arrays)
        ref = native(point)
        from combine_jax.validation import finite_difference
        fd = finite_difference(native,point,np.full(len(point),2e-4))
        report['same_point_native_asimov'] = dict(nll_absolute_error=abs(float(value)-ref),
            gradient_max_absolute_error=float(np.max(abs(np.asarray(grad)-fd))),
            gradient_max_relative_error=float(np.max(abs(np.asarray(grad)-fd)/np.maximum(abs(fd),1e-8))))
        nll.setAnalyticBarlowBeeston(False)
    report['status']='crossed_diagnostic_complete'
    save_json(output,report)
    print(report['status'],report['cases'],report['same_point_native_asimov'])


if __name__=='__main__':
    main()
