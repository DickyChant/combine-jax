"""Bind a shared physical profile grid to revalidated native/device EDM fits."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_tensor_edm_comparison import compare
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.tensor.profile_point import configured_profile_model
from native_fit_model import load_native_fit_model
from tensor_benchmark_common import clean,save_json,sha


def make_plan(*,manifest,case,tensor,native,native_point,parameter,poi,offsets,
              source_revision=None,allow_uncertified_curvature=False):
    # Recompute the complete baseline comparison before reading fit coordinates.
    comparison=compare(tensor,native,native_point,manifest=manifest,case=case,
        source_revision=source_revision,allow_uncertified_curvature=allow_uncertified_curvature)
    if comparison['status']!='passed_same_minimum_gates_no_speed_claim':
        raise ValueError('Profile plans require accepted baseline minima')
    entry=json.loads(Path(manifest).read_text())[case]
    model,_=load_native_fit_model(entry['model']);info=model_parameter_info(model)
    names=info.metadata['names']
    if parameter not in names or poi not in names or parameter==poi:
        raise ValueError('Select distinct active profile parameter and POI')
    selected=names.index(parameter);poi_index=names.index(poi)
    if any(info.arrays['lower'][i]==info.arrays['upper'][i] for i in (selected,poi_index)):
        raise ValueError('Profile parameter and POI must be free')
    offsets=np.asarray(offsets,dtype=np.float64)
    if (offsets.ndim!=1 or len(offsets)<3 or not np.isfinite(offsets).all()
            or len(set(offsets.tolist()))!=len(offsets) or 0. not in offsets
            or not np.any(offsets<0) or not np.any(offsets>0)):
        raise ValueError('Use distinct finite offsets with zero and both signs')
    baseline=json.loads(Path(native).read_text())
    values=np.asarray(baseline['fit']['values']);errors=np.asarray(baseline['fit']['errors'])
    if not np.isfinite(errors[[selected,poi_index]]).all() or np.any(errors[[selected,poi_index]]<=0):
        raise ValueError('Native errors must be positive for the grid and POI shift comparison')
    grid=values[selected]+offsets*errors[selected]
    for value in grid:
        # Also checks native stored ranges; never clip or relabel an endpoint.
        configured_profile_model(model,{parameter:float(value)})
    inputs=dict(manifest=str(manifest),case=case,tensor=str(tensor),native=str(native),native_point=str(native_point),
        parameter=parameter,poi=poi,offsets=offsets.tolist(),source_revision=source_revision,
        allow_uncertified_curvature=allow_uncertified_curvature)
    return clean(dict(status='profile_plan_ready',script_sha256=sha(__file__),inputs=inputs,
        baseline_checker_sha256=sha(Path(__file__).with_name('audit_tensor_edm_comparison.py')),
        artifacts={str(p):sha(p) for p in (manifest,tensor,native,native_point)},
        model_sha256={name:sha(Path(entry['model'])/name) for name in ('model.json','arrays.npz')},
        workspace_sha256=sha(entry['workspace']),dataset=comparison['dataset'],host=baseline['host'],
        parameter_names=names,parameter=parameter,parameter_index=selected,poi=poi,poi_index=poi_index,
        grid=grid,offsets=offsets,initial='imported nominal',initial_values=info.arrays['initial'],
        baseline_values=values,baseline_errors=errors,baseline_canonical_nll=baseline['fit']['canonical_nll'],
        baseline_comparison=comparison,fit_state=info.metadata['fit_policy'],
        point_definition='Native fitted nuisance value plus offset times its Minuit-reported error. Both engines fix the same physical values and start other coordinates at the imported nominal point.',
        scope='Profile/POI-shift comparison plan, not a completed scan. Native errors set grid spacing; offsets do not certify confidence endpoints. HHVV clipping uncertainty limitation remains in force.'))


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('status')!='profile_plan_ready':raise ValueError('Incomplete profile plan')
    for name,digest in plan['artifacts'].items():
        if sha(name)!=digest:raise ValueError('Profile baseline artifact changed')
    expected=make_plan(**plan['inputs'])
    if plan!=expected:raise ValueError('Profile plan differs from its revalidated baseline')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','tensor','native','native-point','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('case','parameter','poi'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--offsets',nargs='+',type=float,default=[-2.,-1.,0.,1.,2.])
    parser.add_argument('--source-revision')
    parser.add_argument('--allow-uncertified-curvature',action='store_true')
    args=parser.parse_args();assert not args.output.exists()
    arguments=vars(args).copy();output=arguments.pop('output')
    result=make_plan(**arguments);save_json(output,result)
    print(result['status'],len(result['grid']),'points',flush=True)


if __name__=='__main__':main()
