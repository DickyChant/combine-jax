"""Complete workspace+card tensor export and 26-point counting NLL/AD gate."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import numpy as np

from combine_jax.root_io import open_workspace,load_caching_nll,REFERENCE_FLAGS
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=Path('reports/tensor_counting.json')
    directory=Path('build/tensor_counting_model')
    assert not output.exists() and not directory.exists(),'Preserve previous evidence'
    started=time.perf_counter()
    manifest_path=Path('build/gpu_exports/manifest_random.json')
    manifest=json.loads(manifest_path.read_bytes())
    reference=manifest['models']['counting']
    workspace_path=Path('build/gpu_exports/counting.root')
    assert sha(workspace_path)==reference['source']['sha256']
    report=dict(status='running',source=reference['source'],manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
        script_sha256=sha(__file__),jax_version=jax.__version__,combine_version=version('higgs-combine-tool'),
        float64=bool(jax.config.x64_enabled),runtime_flags=REFERENCE_FLAGS,cases=[],
        scope='Complete tensor counting-model NLL, gradients and first three Hessians; not the full HHVV model.')
    with open_workspace(workspace_path) as workspace:
        before=time.perf_counter()
        imported=from_workspace(workspace,'tests/cards/counting.txt')
        report['import_seconds']=time.perf_counter()-before
        imported.save(directory)
        model=TensorBundle.load(directory)
        arrays=jax.device_put(dict(model.arrays))
        function=make_model_function(model)
        point=model.arrays['scalars__parameter_initial']
        before=time.perf_counter()
        value=jax.jit(function).lower(point,arrays).compile()
        report['value_lower_compile_seconds']=time.perf_counter()-before
        before=time.perf_counter()
        gradient=jax.jit(jax.grad(function)).lower(point,arrays).compile()
        report['gradient_lower_compile_seconds']=time.perf_counter()-before
        before=time.perf_counter()
        hessian=jax.jit(jax.hessian(function)).lower(point,arrays).compile()
        report['hessian_lower_compile_seconds']=time.perf_counter()-before
        root=load_caching_nll()
        report['root_version']=str(root.gROOT.GetVersion())
        assert report['root_version']==manifest['root_version'] and report['combine_version']==manifest['combine_version']
        native=root.cacheutils.CachingSimNLL(workspace.pdf('model_s'),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        native.clearZeroPoint();native.clearConstantZeroPoint()
        names=model.metadata['parameter_names']
        assert names==reference['reference']['parameters']
        for index,case in enumerate(reference['reference']['cases']):
            theta=np.array(case['theta'])
            for n,v in zip(names,theta,strict=True):workspace.var(n).setVal(float(v))
            nll=float(value(theta,arrays));g=np.array(gradient(theta,arrays))
            ref=float(native.getVal());fd=np.array(case['combine_fd_gradient'])
            relative=np.abs(g-fd)/np.maximum(1e-8,np.abs(fd))
            row=dict(index=index,theta=case['theta'],nll=nll,native_nll=ref,
                nll_absolute_error=abs(nll-ref),numpy_absolute_error=abs(float(evaluate_model_numpy(model,theta))-ref),
                native_saved_absolute_error=abs(ref-case['combine_nll']),gradient=g.tolist(),
                native_fd_gradient=fd.tolist(),gradient_relative_error=relative.tolist(),
                old_ad_gradient_max_error=float(np.max(np.abs(g-np.array(case['jax_gradient'])))))
            if index<3:
                h=np.array(hessian(theta,arrays));previous=np.array(case['hessian']);fdh=np.array(case['fd_hessian'])
                row.update(hessian=h.tolist(),old_ad_hessian_max_error=float(np.max(np.abs(h-previous))),
                           fd_hessian_max_error=float(np.max(np.abs(h-fdh))))
            report['cases'].append(row)
        report.update(parameter_names=names,channels=model.metadata['channel_names'],constraints=model.metadata['constraint_names'],
            structure_key=model.structure_key,artifact=str(directory),
            artifact_hashes={str(p):sha(p) for p in sorted(directory.iterdir())},
            closed_over_arrays=len(jax.make_jaxpr(function)(point,arrays).consts))
    report.update(max_nll_absolute_error=max(c['nll_absolute_error'] for c in report['cases']),
        max_numpy_absolute_error=max(c['numpy_absolute_error'] for c in report['cases']),
        max_gradient_relative_error=max(max(c['gradient_relative_error']) for c in report['cases']),
        max_old_ad_hessian_error=max(c.get('old_ad_hessian_max_error',0.) for c in report['cases']),
        max_fd_hessian_error=max(c.get('fd_hessian_max_error',0.) for c in report['cases']),
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))},
        elapsed_seconds=time.perf_counter()-started)
    passed=(len(report['cases'])==26 and report['max_nll_absolute_error']<1e-6
        and report['max_numpy_absolute_error']<1e-6 and report['max_gradient_relative_error']<1e-4
        and report['max_old_ad_hessian_error']<1e-10 and report['max_fd_hessian_error']<1e-6
        and report['closed_over_arrays']==0)
    report['status']='passed' if passed else 'failed'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','tensor_sources','runtime_flags','artifact_hashes')},indent=2))
    assert passed


if __name__=='__main__':main()
