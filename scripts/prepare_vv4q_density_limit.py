"""Prepare corrected noVH MX4000 background Asimov and capture fixed-point gates."""
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from combine_jax.methods.workspace import prepare_expected_workspace
from combine_jax.root_io import open_workspace, load_root
from combine_jax.tensor.model import from_workspace, make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.data import bind_channels


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=Path('reports/asymptotic_vv4q_noVH_MX4000_density_preparation.json')
    directory=Path('build/asymptotic_vv4q_noVH_MX4000_density')
    assert not out.exists() and not directory.exists()
    source=Path('build/dhi_noVH_MX4000_job/workspace.root')
    card=Path('build/asymptotic_vv4q_inputs/noVH_4000/card.txt')
    data_path=Path('build/vv4q_v10_asimov_bonly/higgsCombineV10Bonly.GenerateOnly.mH125.0.root')
    points=Path('build/vv4q_version_points.json')
    report=dict(status='preparing',script_sha256=sha(__file__),workspace_sha256=sha(source),data_sha256=sha(data_path),
        points_sha256=sha(points),scope='Corrected v10_density model; original v10 workspace, nominal background Asimov only. No observed data fits.',
        sources={str(p):sha(p) for p in Path('src/combine_jax').rglob('*.py')})
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save();start=time.perf_counter()
    model,prep=prepare_expected_workspace(source,card,directory,poi='r_hhvv',run='expected',no_fit_asimov=True,
        parameters={'MX':4000.},freeze=['MX'],mass=125.)
    report.update(preparation=prep,preparation_seconds=time.perf_counter()-start);save()
    root=load_root();handle=root.TFile.Open(str(data_path),'READ');data=handle.Get('toys/toy_asimov')
    with open_workspace(directory/'workspace.root') as ws:
        mc=ws.obj('ModelConfig');category=str(mc.GetPdf().indexCat().GetName())
        channels={ch:('MJ1','MJ2','MJJ') for ch in ('ch4','ch5','ch9','ch10')}
        native_rows=bind_channels(data,category,channels)
        own_rows=bind_channels(ws.data('combine_jax_asimov_bonly'),category,channels)
        checks={}
        for ch in channels:
            left,right=native_rows[ch],own_rows[ch]
            for n in channels[ch]:np.testing.assert_array_equal(left.coordinates[n],right.coordinates[n])
            error=abs(left.weights-right.weights)
            checks[ch]=dict(max_weight_error=float(max(error)),l1_weight_error=float(sum(error)),
                native_sum=float(sum(left.weights)),corrected_sum=float(sum(right.weights)))
        report['generation_vs_v10']=checks;save()
        # The NLL gate uses exactly the archived native-v10 rows. Generated rows
        # are audited separately, so differing data cannot hide a model error.
        getattr(ws,'import')(data,root.RooFit.Rename('v10_shared_asimov'))
        model=from_workspace(ws,card,data='v10_shared_asimov')
        info=model_parameter_info(model);names=info.metadata['names']
        plan=json.loads(points.read_text());order=[plan['parameter_names'].index(n) for n in names]
        function=jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
        arrays=jax.device_put(dict(model.arrays));report['cases']=[];report['parameter_names']=names
        for c in plan['cases']:
            point=np.array(c['point'])[order]
            val,grad=jax.block_until_ready(function(point,arrays))
            report['cases'].append(dict(index=c['index'],nll=float(val)))
            if c['index']==1:report['gradient']=np.asarray(grad).tolist()
            save();print('corrected NLL',c['index'],float(val),flush=True)
    report.update(status='captured',wall_seconds=time.perf_counter()-start);save()


if __name__=='__main__':main()
