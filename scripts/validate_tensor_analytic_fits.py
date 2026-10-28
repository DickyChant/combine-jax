"""Native-objective fit gates for the parametric card and each MultiPdf branch."""
from dataclasses import asdict
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource

import jax
import numpy as np
from iminuit import Minuit

from build_parametric_fixture import build_parametric_fixture
from build_multipdf_fixture import build_multipdf_fixture
from combine_jax.root_io import load_caching_nll,open_workspace
from combine_jax.likelihood import from_workspace as old_import
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.parameter_info import model_parameter_info,pack_parameter_info
from combine_jax.tensor.fitting import prepare_fit


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reports/tensor_analytic_fits.json')
    parser.add_argument('--directory',default='build/tensor_analytic_fits')
    args=parser.parse_args();output=Path(args.output);directory=Path(args.directory)
    assert not output.exists() and not directory.exists()
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    root=load_caching_nll()
    sources={str(p):sha(p) for p in Path('src/combine_jax/tensor').glob('*.py')}
    sources.update({p:sha(p) for p in (__file__,'scripts/build_parametric_fixture.py','scripts/build_multipdf_fixture.py')})
    report=dict(status='running',stage='build',sources=sources,backend=jax.default_backend(),float64=True,
        versions={n:version(n) for n in ('jax','jaxlib','iminuit','higgs-combine-tool')},root_version=str(root.gROOT.GetVersion()),
        reference='Native Combine v11 CachingSimNLL minimized with iminuit; continuous fit with each discrete choice frozen.',
        scope='Numerical fit/covariance/profile validation only. No performance measurement.',
        error_conventions=dict(errors='sqrt(diagonal(raw inverse AD Hessian))',reported_errors='Minuit finite-step bounded reporting; compared to native Minuit errors'),cases=[])
    def save():
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        output.write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    save()
    try:
        parametric=build_parametric_fixture(directory/'parametric')
        multipdf=build_multipdf_fixture(directory/'multipdf')
        for label,path,index in [('parametric',parametric,None),('multipdf_exp',multipdf,0),('multipdf_bern',multipdf,1)]:
            report['stage']=label;save();print('start',label,flush=True)
            with open_workspace(path) as workspace:
                if index is not None:
                    workspace.cat('pdf_index').setIndex(index)
                    for n,e in [('multi_slope',.005),('multi_c1',.1),('multi_c2',.1)]:workspace.var(n).setError(e)
                oracle=old_import(workspace);info=pack_parameter_info(oracle.parameters)
                model=from_workspace(workspace,path.parent/'card.txt')
                imported_info=model_parameter_info(model)
                assert imported_info.metadata==info.metadata
                for n,a in info.arrays.items():np.testing.assert_array_equal(imported_info.arrays[n],a)
                artifact=directory/(label+'_model');model.save(artifact)
                native=root.cacheutils.CachingSimNLL(workspace.obj('ModelConfig').GetPdf(),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
                native.setAnalyticBarlowBeeston(True);native.clearConstantZeroPoint();native.clearZeroPoint()
                def objective(*point):
                    for n,v in zip(oracle.parameter_names,point,strict=True):workspace.var(n).setVal(float(v))
                    return float(native.getVal())
                ref=Minuit(objective,*oracle.initial,name=oracle.parameter_names);ref.errordef=.5;ref.tol=1e-8;ref.strategy=2
                for p in oracle.parameters:
                    ref.limits[p.name]=(p.minimum,p.maximum);ref.errors[p.name]=p.error if p.error>0 else .1
                ref.migrad();ref.hesse()
                nv=np.array(ref.values);ne=np.array(ref.errors)
                case=dict(label=label,status='fitting',workspace=str(path),workspace_sha256=sha(path),card_sha256=sha(path.parent/'card.txt'),
                    model=str(artifact),model_sha256={n:sha(artifact/n) for n in ('model.json','arrays.npz')},
                    parameters=[asdict(p) for p in oracle.parameters],discrete_choices=oracle.discrete_choices,
                    native=dict(valid=bool(ref.valid),accurate_covariance=bool(ref.fmin.has_accurate_covar),edm=float(ref.fmin.edm),
                        values=nv,errors=ne,nll=float(ref.fval),covariance=np.array(ref.covariance) if ref.covariance is not None else None,
                        has_parameters_at_limit=bool(ref.fmin.has_parameters_at_limit)),profiles=[])
                report['cases'].append(case);save()
                fn,c,a=prepare_fit(model,gradient_tolerance=1e-7)
                result=jax.device_get(fn(c,a))
                case['device']={k:result[k] for k in ('status','covariance_status','iteration','value','values','errors','reported_errors','reported_error_status','covariance','hessian','active_bounds','projected_gradient','inverse_residual','roundoff_accepts','maximum_roundoff_allowance')}
                case['status']='checking';save()
                failures=[]
                if not ref.valid or not ref.fmin.has_accurate_covar:failures.append('native_fit_or_covariance_invalid')
                if result['status']!=0:failures.append('device_not_converged')
                if result['covariance_status']!=0:failures.append('device_covariance_invalid')
                nll_error=abs(float(result['value'])-float(ref.fval));case['nll_absolute_error']=nll_error
                if not nll_error<1e-6:failures.append('nll')
                parameter_tolerance=np.maximum(.01*np.abs(nv),.01*ne)
                parameter_ratio=np.abs(result['values']-nv)/np.maximum(parameter_tolerance,1e-30)
                case['parameter_error_over_tolerance']=parameter_ratio
                if not np.all(parameter_ratio<1.):failures.append('parameters')
                error_ratio=np.abs(result['reported_errors']-ne)/np.maximum(ne,1e-30);case['error_relative_difference']=error_ratio
                case['curvature_error_vs_native_reported_relative_difference']=np.abs(result['errors']-ne)/np.maximum(ne,1e-30)
                if not np.all(result['reported_error_status']==0):failures.append('reported_error_status')
                if not np.all(error_ratio<.01):failures.append('uncertainties')
                if ref.covariance is None or not np.allclose(result['covariance'],np.array(ref.covariance),atol=1e-5,rtol=.01):failures.append('covariance')
                if result['status']==0 and result['covariance_status']==0:
                    old_h=np.asarray(oracle.hessian(result['values']))
                    case['old_hessian_max_absolute_error']=float(np.max(np.abs(result['hessian']-old_h)))
                    if not np.allclose(result['hessian'],old_h,atol=1e-8,rtol=1e-9):failures.append('old_hessian')
                case['failed_gates']=failures;save()
                if not failures:
                    poi=oracle.parameter_names.index('r')
                    for endpoint in result['values'][poi]+np.array([-.5,.5])*result['errors'][poi]:
                        if not info.arrays['lower'][poi]<endpoint<info.arrays['upper'][poi]:
                            failures.append('profile_endpoint_outside_bounds');break
                        pf,pc,pa=prepare_fit(model,initial=result['values'],fixed={'r':float(endpoint)},gradient_tolerance=1e-7)
                        p=jax.device_get(pf(pc,pa))
                        ref.values=nv;ref.values['r']=endpoint;ref.fixed['r']=True;ref.migrad()
                        error=abs(float(p['value'])-float(ref.fval))
                        agree=bool(ref.valid and p['status']==0 and error<1e-6 and np.allclose(p['values'],np.array(ref.values),atol=1e-4,rtol=1e-4))
                        case['profiles'].append(dict(r=float(endpoint),passed=agree,device_status=int(p['status']),native_valid=bool(ref.valid),
                            projected_gradient=p['projected_gradient'],iterations=int(p['iteration']),roundoff_accepts=int(p['roundoff_accepts']),
                            maximum_roundoff_allowance=float(p['maximum_roundoff_allowance']),
                            device_nll=float(p['value']),native_nll=float(ref.fval),nll_absolute_error=error,
                            device_values=p['values'],native_values=np.array(ref.values)))
                        if not agree:failures.append('profile')
                        save()
                case['status']='passed' if not failures else 'failed';case['failed_gates']=failures
                native.setAnalyticBarlowBeeston(False);save()
                print(label,case['status'],failures,'NLL error',nll_error,flush=True)
                jax.clear_caches()
        assert all(sha(p)==h for p,h in sources.items())
        report['status']='passed' if all(c['status']=='passed' for c in report['cases']) else 'failed_gates'
        report['stage']='complete';save()
    except BaseException as error:
        report['status']='error';report['error']=repr(error);save();raise


if __name__=='__main__':main()
