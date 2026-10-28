"""Apply unchanged NLL/gradient gates to the corrected v10-density import."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    actual_path=Path('reports/asymptotic_vv4q_noVH_MX4000_density_preparation.json')
    native_path=Path('reports/asymptotic_vv4q_v10_density_gradient_reference.json')
    out=Path('reports/asymptotic_vv4q_noVH_MX4000_density_gate.json')
    assert not out.exists()
    actual=json.loads(actual_path.read_text()); native=json.loads(native_path.read_text())
    assert actual['status']=='captured' and native['status']=='completed'
    for key in ('workspace_sha256','data_sha256','points_sha256'):
        assert actual[key]==native[key]
    assert len(actual['cases'])==len(native['cases'])==26
    errors=[]
    for a,b in zip(actual['cases'],native['cases'],strict=True):
        assert a['index']==b['index'];errors.append(a['nll']-b['nll'])
    names=actual['parameter_names']; ref=native['gradient']; assert ref['point_index']==1
    fd=np.array([ref['values'][ref['parameter_names'].index(n)] for n in names])
    ad=np.array(actual['gradient']); delta=abs(ad-fd); relative=delta/np.maximum(abs(fd),1e-8)
    generated_exact=all(c['max_weight_error']==0 for c in actual['generation_vs_v10'].values())
    report=dict(status='passed' if max(map(abs,errors))<1e-6 and max(relative)<1e-4 and generated_exact else 'failed',
        script_sha256=sha(__file__),inputs={str(p):sha(p) for p in (actual_path,native_path)},
        reference='Combine v10.3.3 / CMSSW_14_1_0_pre4 / ROOT 6.30.07',template_convention='v10_density',
        nll_tolerance=1e-6,gradient_relative_tolerance=1e-4,nll_deltas=errors,max_nll_absolute_error=max(map(abs,errors)),
        max_gradient_absolute_error=float(max(delta)),max_gradient_relative_error=float(max(relative)),
        largest_gradient_relative_parameter=names[int(np.argmax(relative))],
        generated_weights_identical_to_native_v10=generated_exact,
        scope='Nominal plus 25 fixed random points on identical native-v10 background Asimov rows; full five-point FD gradient at point 1; independent generated-row equality.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
    assert report['status']=='passed'


if __name__=='__main__':main()
