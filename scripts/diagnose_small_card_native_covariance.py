"""Separate native HESSE matrix accuracy from actual objective curvature."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import jax
import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    root=load_caching_nll()
    assert root.gROOT.GetVersion()=='6.40.00' and version('higgs-combine-tool')=='11.0.0'
    manifest_path=Path('build/tensor_benchmark_inputs/manifest.json')
    manifest=json.loads(manifest_path.read_text())
    report=dict(provenance(),status='checking',script_sha256=sha(__file__),float64=True,
        jax_version=jax.__version__,root_version=str(root.gROOT.GetVersion()),combine_version='11.0.0',
        manifest_sha256=sha(manifest_path),cases=[],
        native_reference_policy='Symmetric second differences at five predefined step fractions of saved native errors: .08,.04,.02,.01,.005. Richardson extrapolation cancels leading O(h^2) error. Accept the finest extrapolate only when its difference from the preceding extrapolate is below 1e-6 in Hessian entries scaled by the native errors. Reference construction and step selection do not use AD.',
        scope='Objective curvature at the exact saved native CLI fit point. Does not change fitted points, default HESSE matrices, AD Hessians, or acceptance tolerances; no matrix repair.')
    monitor=Resources(False,args.output)
    def save():
        report['resources']=monitor.snapshot();save_json(args.output,report)
    save()
    try:
        for label in ('counting','automcstats'):
            case=manifest[label]
            model=TensorBundle.load(case['model']);info=model_parameter_info(model)
            native_path=Path(f'reports/gpu_tensor_local_{label}_combine_fit_parser_v4.json')
            saved=json.loads(native_path.read_text());fit=saved['fit']
            assert saved['status']=='measured_fit_pending_agreement' and fit['status']==0 and fit['covariance_quality']==3
            assert fit['parameters']==info.metadata['names']
            point=np.array(fit['values']);scale=np.array(fit['errors']);cli_cov=np.array(fit['covariance'])
            assert np.all(scale>0) and np.all(info.arrays['lower']<info.arrays['upper'])
            assert saved['input_workspace_sha256']==sha(case['workspace'])==case['hashes'][case['workspace']]
            row=dict(case=label,native_fit=str(native_path),native_fit_sha256=sha(native_path),
                parameters=info.metadata['names'],point=point,step_scale=scale,steps=[],
                native_cli_covariance=cli_cov,model_sha256={n:sha(Path(case['model'])/n) for n in ('model.json','arrays.npz')})
            report['cases'].append(row);save()
            with open_workspace(case['workspace']) as workspace:
                for name,configuration in case['configuration'].items():
                    workspace.var(name).setVal(configuration['value']);workspace.var(name).setConstant(configuration['constant'])
                variables=[workspace.var(n) for n in info.metadata['names']]
                config=workspace.obj('ModelConfig')
                nll=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('data_obs'),config.GetNuisanceParameters())
                nll.setAnalyticBarlowBeeston(True);nll.clearConstantZeroPoint();nll.clearZeroPoint()
                root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
                evaluations=0
                def evaluate(theta):
                    nonlocal evaluations
                    assert np.all(theta>=info.arrays['lower']) and np.all(theta<=info.arrays['upper'])
                    for variable,value in zip(variables,theta,strict=True):variable.setVal(float(value))
                    root.RooAbsReal.clearEvalErrorLog()
                    value=float(nll.getVal());evaluations+=1
                    assert np.isfinite(value) and root.RooAbsReal.numEvalErrors()==0
                    return value
                row['native_canonical_nll']=evaluate(point)
                nll.setZeroPoint()
                center=evaluate(point)
                n=len(point)
                for fraction in (.08,.04,.02,.01,.005):
                    step=fraction*scale
                    matrix=np.empty((n,n))
                    for i in range(n):
                        di=np.zeros(n);di[i]=step[i]
                        matrix[i,i]=(evaluate(point+di)-2*center+evaluate(point-di))/step[i]**2
                        for j in range(i):
                            dj=np.zeros(n);dj[j]=step[j]
                            matrix[i,j]=matrix[j,i]=(evaluate(point+di+dj)-evaluate(point+di-dj)
                                -evaluate(point-di+dj)+evaluate(point-di-dj))/(4*step[i]*step[j])
                    row['steps'].append(dict(fraction=fraction,physical_steps=step,hessian=matrix));save()
                row['zero_point_restoration_error']=abs(evaluate(point)-center)
                row['native_evaluations']=evaluations
                assert row['zero_point_restoration_error']<1e-10
                nll.setAnalyticBarlowBeeston(False)
            extrapolates=[(4*right['hessian']-left['hessian'])/3 for left,right in zip(row['steps'][:-1],row['steps'][1:])]
            outer_scale=scale[:,None]*scale[None,:]
            stability=float(np.max(np.abs((extrapolates[-1]-extrapolates[-2])*outer_scale)))
            reference=extrapolates[-1]
            row.update(richardson_hessians=extrapolates,native_only_scaled_stability=stability,
                       native_reference_stable=stability<1e-6,native_reference_hessian=reference)
            assert row['native_reference_stable']
            assert np.linalg.eigvalsh(reference).min()>0
            reference_cov=np.linalg.inv(reference)
            arrays=jax.device_put({n:a for n,a in model.arrays.items() if not n.startswith('parameter_info__')})
            function=make_model_function(model)
            value,gradient=jax.device_get(jax.jit(jax.value_and_grad(function))(point,arrays))
            ad_hessian=np.array(jax.jit(jax.hessian(function))(point,arrays))
            ad_covariance=np.linalg.inv(ad_hessian)
            error=float(np.max(np.abs((ad_hessian-reference)*outer_scale)))
            ad_pass=bool(error<1e-6 and np.allclose(ad_covariance,reference_cov,rtol=.01,atol=1e-5))
            cli_pass=bool(np.allclose(cli_cov,reference_cov,rtol=.01,atol=1e-5))
            row.update(ad_nll=float(value),native_nll_absolute_error=abs(float(value)-row['native_canonical_nll']),
                ad_gradient=gradient,ad_hessian=ad_hessian,ad_covariance=ad_covariance,native_reference_covariance=reference_cov,
                ad_vs_native_scaled_hessian_max_error=error,ad_agrees_with_native_curvature=ad_pass,
                default_cli_covariance_agrees_with_native_curvature=cli_pass,
                default_cli_covariance_identity_residual=float(np.max(np.abs(reference@cli_cov-np.eye(n)))),
                ad_covariance_identity_residual=float(np.max(np.abs(reference@ad_covariance-np.eye(n)))))
            assert ad_pass and row['native_nll_absolute_error']<1e-6
            save();jax.clear_caches()
            print(label,'stable native FD',stability,'AD error',error,'CLI covariance accepted',cli_pass,flush=True)
        report.update(status='passed_native_curvature_diagnostic',conclusion='Both native-only finite-difference sequences are stable and agree with raw AD curvature/inverses at the saved native points. Default CLI HESSE covariance discrepancies are retained and quantified against that same native curvature; no default-CLI matrix-agreement claim.')
        assert all(sha(p)==digest for p,digest in report['sources'].items());save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        monitor.close()


if __name__=='__main__':
    main()
