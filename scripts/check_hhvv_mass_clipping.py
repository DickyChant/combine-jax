"""Independently reconstruct the four probed bins and native normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from audit_hhvv_mass_clipping import CPP
from tensor_benchmark_common import save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); assert not args.output.exists()
    report=json.loads(args.report.read_text())
    assert report['status']=='native_snapshot_probes_and_template_comparison_complete'
    assert report['script_sha256']==sha('scripts/audit_hhvv_mass_clipping.py')
    for entry in report['workspaces'].values(): assert sha(entry['path'])==entry['sha256']
    left,right=(report['inventories'][str(m)]['templates'] for m in (2000,4000))
    assert len(left)==len(right)==1032 and left==right
    prior=json.loads(Path(report['prior_audit']).read_text()); assert sha(report['prior_audit'])==report['prior_audit_sha256']
    scales={}
    for p,digest in prior['native_probes'].items():
        assert sha(p)==digest
        for row in json.loads(Path(p).read_text())['points']: scales[row['name']]=row['scale']
    from combine_jax.root_io import load_root,open_workspace
    root=load_root(); assert root.gInterpreter.Declare(CPP)
    comparisons=[]
    with open_workspace(report['workspaces']['4000']['path']) as workspace:
        workspace.loadSnapshot('MultiDimFit')
        for row in report['probes']:
            pdf=workspace.pdf(row['histogram']); view=root.mass_clipping_audit.View2D(pdf)
            nominal=pdf.cacheNominal(); shape=(nominal.binX(),nominal.binY())
            raw=np.array(root.mass_clipping_audit.values(nominal))
            assert hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest()==right[row['histogram']]['buffers']['nominal']['sha256']
            region=pdf.smoothRegion(); assert region>0 and pdf.smoothAlgo()>=0
            for coefficient,morph in zip(pdf.coefList(),pdf.morphs(),strict=True):
                theta=coefficient.getVal(); x=theta/region
                alpha=np.sign(theta) if abs(theta)>=region else x*(x*x*(3*x*x-10)+15)/8
                raw+=.5*theta*(np.array(root.mass_clipping_audit.values(morph.diff))+alpha*np.array(root.mass_clipping_audit.values(morph.sum)))
            cpp=np.array(view.raw()); np.testing.assert_allclose(raw,cpp,rtol=0,atol=1e-15)
            widths=np.array([nominal.GetWidth(i) for i in range(nominal.fullsize())]).reshape(shape)
            expected=np.maximum(raw,1e-9).reshape(shape)
            axis=1 if pdf.conditional() else (0,1)
            expected/=np.sum(expected*widths,axis=axis,keepdims=True)
            native=np.array(view.normalized()).reshape(shape)
            np.testing.assert_allclose(expected,native,rtol=1e-12,atol=1e-15)
            k=row['flat_bin']; assert cpp[k]==row['raw_bin'] and cpp[k]>1e-9
            assert row['scale']==scales[row['parameter']]
            assert [p['step'] for p in row['probes']]==[f*row['scale'] for f in report['step_fractions']]
            assert all(p['restoration_absolute_error']==0 for p in row['probes'])
            comparisons.append(dict(histogram=row['histogram'],flat_bin=k,raw_bin=float(cpp[k]),
                raw_over_floor=float(cpp[k]/1e-9),native_raw_numpy_max_error=float(np.max(abs(raw-cpp))),
                native_normalized_numpy_max_error=float(np.max(abs(expected-native))),
                steps_crossing_floor=sum(p['crosses_floor'] for p in row['probes']),
                steps_bracketing_zero=sum(p['signs_bracket_zero'] for p in row['probes'])))
    assert report['canonical_restoration_absolute_error']==0
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    save_json(args.output,dict(status='passed_selected_bins_not_clipped_templates_identical',
        script_sha256=sha(__file__),report=str(args.report),report_sha256=sha(args.report),
        templates_compared=1032,template_classes=dict(__import__('collections').Counter(r['class_name'] for r in right.values())),
        point=report['snapshot'],comparisons=comparisons,
        conclusion='All four selected bins exceed the 1e-9 floor at the saved MX4000 MultiDimFit point; the expectation of the same four clipped bins is refuted at that point. All 1032 WW/bbWW template objects have matching exact float64 bytes and interpolation metadata. This is not a fit/covariance certificate or a claim that no other MX4000 bins clip.'))
    print('passed: 1032 identical templates; 0/4 selected MX4000 bins clipped')


if __name__=='__main__': main()
