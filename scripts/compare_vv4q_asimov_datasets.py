"""Compare native v11 pre-fit Asimov rows for the two VV4q workspace layouts."""
import time
from pathlib import Path
import numpy as np
from tensor_benchmark_common import save_json, sha
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.dataset_identity import workspace_dataset_identity
from combine_jax.data import bind_channels


def bindings(ws, data):
    mc = ws.obj('ModelConfig')
    pdf = mc.GetPdf()
    channels = {}
    for state in pdf.indexCat():
        name = str(state.first)
        channels[name] = sorted(str(v.GetName()) for v in pdf.getPdf(name).getObservables(mc.GetObservables())
                                if v.InheritsFrom('RooAbsReal'))
    return bind_channels(data,str(pdf.indexCat().GetName()),channels)


def main():
    output = Path('reports/asymptotic_vv4q_noVH_MX4000_asimov_comparison.json')
    assert not output.exists()
    prepared = Path('build/asymptotic_vv4q_noVH_MX4000/workspace.root')
    source = Path('build/dhi_noVH_MX4000_job/workspace.root')
    start = time.perf_counter()
    report = dict(status='running',script_sha256=sha(__file__),local_workspace_sha256=sha(prepared),
        dhi_workspace_sha256=sha(source), scope='Generate background pre-fit Asimov only; no fitting or observed dataset access. Both use native v11 utilities in ROOT 6.40.')
    save_json(output,report)
    root = load_caching_nll()
    root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"')
    with open_workspace(prepared) as ws:
        dataset = ws.data('combine_jax_asimov_bonly')
        report['local_identity'] = workspace_dataset_identity(ws,dataset)
        a = bindings(ws,dataset)
    with open_workspace(source) as ws:
        ws.var('MX').setRange(1000.,6000.)
        ws.var('MX').setVal(4000.)
        assert ws.var('MX').isConstant()
        mc = ws.obj('ModelConfig')
        generated = root.asimovutils.asimovDatasetNominal(mc,0.,-1)
        report['dhi_identity'] = workspace_dataset_identity(ws,generated)
        b = bindings(ws,generated)
    report['channels'] = {}
    for name in sorted(a):
        aa, bb = a[name], b[name]
        coords_match = all(np.array_equal(aa.coordinates[n],bb.coordinates[n]) for n in aa.coordinates)
        error = abs(aa.weights-bb.weights)
        report['channels'][name] = dict(coordinates_equal=coords_match,rows=len(aa.weights),
            local_sum_weights=float(sum(aa.weights)),dhi_sum_weights=float(sum(bb.weights)),
            max_weight_absolute_error=float(max(error)),max_weight_relative_error=float(max(error/np.maximum(abs(bb.weights),1e-300))))
    report.update(status='completed',identical=report['local_identity']==report['dhi_identity'],wall_seconds=time.perf_counter()-start)
    save_json(output,report)
    print(report['identical'],report['channels'],flush=True)


if __name__=='__main__':
    main()
