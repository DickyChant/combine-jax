"""Native same-point and Asimov-generation audit for the noVH MX4000 limit."""
import json
from pathlib import Path
import time

import jax
import numpy as np
from tensor_benchmark_common import save_json, sha
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.dataset_identity import workspace_dataset_identity
from combine_jax.validation import finite_difference


def main():
    output = Path('reports/asymptotic_vv4q_noVH_MX4000_likelihood_audit.json')
    assert not output.exists()
    base = Path('build/asymptotic_vv4q_noVH_MX4000')
    model = TensorBundle.load(base/'model')
    info = model_parameter_info(model)
    names = info.metadata['names']
    report = dict(status='running',script_sha256=sha(__file__),runtime_flags=REFERENCE_FLAGS,
        workspace_sha256=sha(base/'workspace.root'), model_sha256=sha(base/'model/model.json'),
        source_files={str(p):sha(p) for p in Path('src/combine_jax/tensor').glob('*.py')},
        parameter_names=names,dataset=model.metadata['asymptotic_asimov'],
        scope='Background pre-fit Asimov only. Native v11 same-point NLL at nominal + 25 random points; full finite-difference gradient at one random point. Independently compare Asimov generation from the original dhi workspace.',
        cases=[])
    save_json(output,report)
    start = time.perf_counter()
    root = load_caching_nll()
    arrays = jax.device_put(dict(model.arrays))
    initial = info.arrays['initial'].copy()
    function = jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
    value, gradient = jax.block_until_ready(function(initial,arrays))
    report['nominal_nll'] = float(value)
    previous = json.loads(Path('reports/asymptotic_vv4q_noVH_MX4000.json').read_text())
    report['reported_global_minus_nominal_nll'] = previous['profiles'][0]['value']-float(value)
    report['compile_seconds'] = time.perf_counter()-start
    save_json(output,report)
    with open_workspace(base/'workspace.root') as ws:
        mc = ws.obj('ModelConfig')
        dataset = ws.data('combine_jax_asimov_bonly')
        nll = root.cacheutils.CachingSimNLL(mc.GetPdf(),dataset,mc.GetNuisanceParameters())
        nll.clearConstantZeroPoint(); nll.clearZeroPoint()
        variables = [ws.var(n) for n in names]
        def native(point):
            for v,x in zip(variables,point,strict=True):v.setVal(float(x))
            return float(nll.getVal())
        rng = np.random.default_rng(7319)
        scale = np.where(info.arrays['error']>0,info.arrays['error'],1.)
        scale[names.index('r_hhvv')] = 1.
        lower,upper = info.arrays['lower'],info.arrays['upper']
        margin = np.minimum(.01*scale,.01*(upper-lower))
        points = [initial]+[np.clip(initial+rng.uniform(-2,2,len(initial))*scale,lower+margin,upper-margin) for _ in range(25)]
        points[1][names.index('r_hhvv')] = 3.631591796875
        for i,point in enumerate(points):
            result = function(point,arrays)
            ref = native(point)
            report['cases'].append(dict(index=i,point=point.tolist(),jax_nll=float(result[0]),native_nll=ref,
                nll_absolute_error=abs(float(result[0])-ref)))
            print('NLL',i,report['cases'][-1]['nll_absolute_error'],flush=True)
            save_json(output,report)
        point = points[1]
        baseline = native(point)
        fd = finite_difference(lambda x:native(x)-baseline,point,2e-4*scale,bounds=(lower,upper))
        ad = np.asarray(function(point,arrays)[1])
        error = abs(ad-fd)
        report['gradient'] = dict(point_index=1,steps=(2e-4*scale).tolist(),jax=ad.tolist(),native_fd=fd.tolist(),
            max_absolute_error=float(max(error)),max_relative_error=float(max(error/np.maximum(abs(fd),1e-8))))
        native(initial)
        save_json(output,report)
        # Compare preparation independently of tensor evaluation and fitting.
        root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"')
        with open_workspace('build/dhi_noVH_MX4000_job/workspace.root') as other:
            other.var('MX').setVal(4000.)
            other.var('MX').setRange(1000.,6000.)
            assert other.var('MX').isConstant()
            cm = other.obj('ModelConfig')
            generated = root.asimovutils.asimovDatasetNominal(cm,0.,-1)
            identity = workspace_dataset_identity(other,generated)
            report['dhi_workspace_v11_asimov'] = dict(identity=identity,
                matches_local=identity==model.metadata['asymptotic_asimov']['dataset_identity'],
                sum_weights=float(generated.sumEntries()),local_sum_weights=float(dataset.sumEntries()))
        report['max_nll_absolute_error'] = max(x['nll_absolute_error'] for x in report['cases'])
        report['nll_passed'] = report['max_nll_absolute_error'] < 1e-6
        report['gradient_passed'] = report['gradient']['max_relative_error'] < 1e-4
        report.update(status='completed',wall_seconds=time.perf_counter()-start)
        save_json(output,report)
    print(report['status'],report['nll_passed'],report['gradient_passed'],flush=True)


if __name__=='__main__':
    main()
