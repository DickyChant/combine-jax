"""Record the native RooFit/CachingAddNLL negated-exponential discrepancy."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from combine_jax.root_io import load_caching_nll,REFERENCE_FLAGS


def main():
    output=Path('reports/tensor_negated_exponential_native_probe.json');assert not output.exists()
    root=load_caching_nll();root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors);root.RooAbsReal.clearEvalErrorLog()
    x=root.RooRealVar('probe_x','',0.,-4.,5.);slope=root.RooRealVar('probe_slope','',0.,-.4,.3)
    rate=root.RooRealVar('probe_rate','',12.,2.,30.);weight=root.RooRealVar('probe_weight','',1.,0.,100.)
    columns=root.RooArgSet(x,weight);data=root.RooDataSet('probe_data','',columns,root.RooFit.WeightVar(weight))
    coordinates=np.linspace(-4.,5.,9)[1:-1];weights=np.array([0.]+[.2+.3*i for i in range(1,len(coordinates))])
    for v,w in zip(coordinates,weights,strict=True):x.setVal(float(v));data.add(columns,float(w))
    cases=[];held=[];norm=root.RooArgSet(x);observed=float(np.sum(weights));extended=12.-observed-observed*np.log(12./observed)
    for negate in (False,True):
        shape=root.RooExponential(f'probe_shape_{negate}','',x,slope,negate)
        pdf=root.RooAddPdf(f'probe_sum_{negate}','',root.RooArgList(shape),root.RooArgList(rate))
        native=root.cacheutils.CachingAddNLL(f'probe_nll_{negate}','',pdf,data,True);native.clearZeroPoint();native.clearConstantZeroPoint()
        held.extend([shape,pdf,native])
        def evaluate(t):
            slope.setVal(float(t));cached=float(native.getVal());densities=[]
            for value in coordinates:x.setVal(float(value));densities.append(shape.getVal(norm))
            direct=float(-np.dot(weights,np.log(densities))+extended)
            return cached,direct
        points=[dict(slope=t,caching_nll=evaluate(t)[0],roofit_density_nll=evaluate(t)[1]) for t in (-.2,0.,.2)]
        upper=np.array(evaluate(1e-4));lower=np.array(evaluate(-1e-4));gradient=(upper-lower)/2e-4
        cases.append(dict(negate=negate,points=points,nominal_fd_caching_gradient=float(gradient[0]),nominal_fd_roofit_gradient=float(gradient[1])))
    assert all(abs(q['caching_nll']-q['roofit_density_nll'])<1e-10 for q in cases[0]['points'])
    assert abs(cases[1]['nominal_fd_caching_gradient']-cases[1]['nominal_fd_roofit_gradient'])>20
    assert root.RooAbsReal.numEvalErrors()==0
    source=Path('vendor/combine_pypi/vendor/CombinedLimit/src/VectorizedSimplePdfs.cc')
    report=dict(status='native_mismatch_confirmed',combine_version=version('higgs-combine-tool'),root_version=str(root.gROOT.GetVersion()),
        runtime_flags=REFERENCE_FLAGS,cases=cases,coordinates=coordinates.tolist(),weights=weights.tolist(),root_evaluation_errors=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_source=str(source),native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        disposition='The tensor channel importer rejects negated RooExponential explicitly. No altered native flag or substituted sign is used.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],cases[1]['nominal_fd_caching_gradient'],cases[1]['nominal_fd_roofit_gradient'])


if __name__=='__main__':main()
