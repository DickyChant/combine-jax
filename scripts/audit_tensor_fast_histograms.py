"""Independently rederive saved bin/integral errors and verify evidence bindings."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_fast_histograms.json');report=json.loads(path.read_bytes())
    output=Path('reports/tensor_fast_histograms_audit.json');assert not output.exists()
    assert report['status']=='passed' and report['float64'] and report['backend']=='cpu'
    assert sha('scripts/validate_tensor_fast_histograms.py')==report['script_sha256']
    for p,h in report['tensor_sources'].items():assert sha(p)==h
    for p,h in report['artifacts'].items():assert sha(p)==h
    for field in ('validation_arrays','native_points_report','scalar_report'):assert sha(report[field])==report[field+'_sha256']
    native=json.loads(Path(report['native_points_report']).read_bytes())
    assert native['source']==report['source'] and report['closed_over_arrays']==0
    with np.load(report['validation_arrays'],allow_pickle=False) as archive:
        points=archive['points'];names=archive['names'];offsets=archive['offsets'];sizes=archive['sizes']
        np.testing.assert_array_equal(points,np.array([c['theta'] for c in native['cases']]))
        assert len(points)==26 and len(names)==731 and len(set(names))==731
        np.testing.assert_array_equal(offsets,np.r_[0,np.cumsum(sizes)[:-1]])
        assert int(np.sum(sizes))==1468354==report['bins']
        results={}
        for kind,count in [('bins',1468354),('integrals',731)]:
            reference=archive['native_'+kind]
            assert reference.shape==(26,count) and np.all(np.isfinite(reference))
            for backend in ('jax','numpy'):
                actual=archive[backend+'_'+kind]
                assert actual.shape==reference.shape and np.all(np.isfinite(actual))
                field=kind if backend=='jax' else backend+'_'+kind
                maximum_absolute=maximum_relative=maximum_ratio=0.
                for i,(a,r) in enumerate(zip(actual,reference,strict=True)):
                    delta=np.abs(a-r);tolerance=report['tolerances']['absolute']+report['tolerances']['relative']*np.abs(r)
                    absolute=float(np.max(delta));relative=float(np.max(delta/np.maximum(np.abs(r),1e-300)));ratio=float(np.max(delta/tolerance))
                    saved=report['cases'][i][field]
                    assert absolute==saved['max_absolute_error'] and relative==saved['max_relative_error'] and ratio==saved['max_tolerance_ratio']
                    assert ratio<=1 and report['cases'][i]['root_evaluation_errors']==0
                    maximum_absolute=max(maximum_absolute,absolute);maximum_relative=max(maximum_relative,relative);maximum_ratio=max(maximum_ratio,ratio)
                results[backend+'_'+kind]=dict(shape=list(actual.shape),max_absolute_error=maximum_absolute,
                    max_relative_error=maximum_relative,max_tolerance_ratio=maximum_ratio)
                del actual
            del reference
    geometry=json.loads(Path('build/tensor_fast_histograms_validation/histograms/model.json').read_bytes())['metadata']
    assert geometry['names']==names.tolist() and report['folded_histograms']==len(geometry['folded_histograms'])
    assert report['batches']==len(geometry['schedule'])
    # Rebuild geometric integral weights independently of exported query maps.
    with np.load('build/tensor_fast_histograms_validation/histograms/arrays.npz',allow_pickle=False) as buffers:
        with np.load(report['validation_arrays'],allow_pickle=False) as archive:
            bins=archive['native_bins'];integrals=archive['native_integrals']
            for j,h in enumerate(geometry['histograms']):
                edges=[buffers[k] for k in h['edges']];bounds=buffers[h['bounds']]
                spans=[np.maximum(0.,np.minimum(e[1:],hi)-np.maximum(e[:-1],lo)) for e,(lo,hi) in zip(edges,bounds,strict=True)]
                volume=np.ones(h['shape'])
                for d,w in enumerate(spans):
                    shape=[1]*len(spans);shape[d]=len(w);volume*=w.reshape(shape)
                expected=np.sum(bins[:,h['offset']:h['offset']+h['size']]*volume.reshape(-1),axis=1)
                np.testing.assert_array_equal(expected,integrals[:,j])
    testlog=Path('reports/tensor_fast_histograms_tests.log');assert '64 passed, 2 warnings' in testlog.read_text()
    baseline='8c8cbdd';unchanged={}
    for p in subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'src/combine_jax'],text=True).splitlines():
        if p.startswith('src/combine_jax/tensor/'):continue
        assert Path(p).read_bytes()==subprocess.check_output(['git','show',baseline+':'+p]);unchanged[p]=sha(p)
    result=dict(status='passed',report=str(path),report_sha256=sha(path),script_sha256=sha(__file__),results=results,
        histograms=731,grid_bins=1468354,points=26,batches=report['batches'],folded_histograms=report['folded_histograms'],
        native_geometric_integrals_reconstructed=True,closed_over_arrays=0,tests=dict(passed=64,warnings=2,log=str(testlog),sha256=sha(testlog)),
        old_backend_baseline=baseline,old_backend_unchanged_files=unchanged,
        scope='Component bin/integral gate only. Complete tensor HHVV channel coverage remains 360/459; full fits, performance and persistent compilation reuse remain pending.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='old_backend_unchanged_files'},indent=2))


if __name__=='__main__':main()
