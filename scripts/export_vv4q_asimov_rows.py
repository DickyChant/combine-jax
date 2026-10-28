"""Read only the two identified background Asimov datasets for NumPy audits."""
from pathlib import Path
import numpy as np
from compare_vv4q_asimov_datasets import bindings
from combine_jax.root_io import load_root,open_workspace
from tensor_benchmark_common import sha,save_json


def main():
    root=load_root();arrays={};sources={}
    workspace='build/asymptotic_vv4q_noVH_MX4000/workspace.root'
    with open_workspace(workspace) as ws:
        for label,path in [('v10','build/vv4q_v10_asimov_bonly/higgsCombineV10Bonly.GenerateOnly.mH125.0.root'),
                           ('v11','build/vv4q_v11_cli_asimov_bonly/higgsCombineV11Bonly.GenerateOnly.mH125.0.root')]:
            sources[path]=sha(path)
            handle=root.TFile.Open(path,'READ');data=handle.Get('toys/toy_asimov')
            assert data
            for channel,binding in bindings(ws,data).items():
                arrays[label+'__'+channel+'__weights']=binding.weights.copy()
                for name,values in binding.coordinates.items():arrays[label+'__'+channel+'__'+name]=values.copy()
            handle.Close()
    output=Path('build/vv4q_asimov_generation_rows.npz')
    np.savez_compressed(output,**arrays)
    save_json(Path('reports/asymptotic_vv4q_asimov_row_export.json'),dict(status='completed',
        script_sha256=sha(__file__),workspace_sha256=sha(workspace),sources=sources,
        arrays=str(output),arrays_sha256=sha(output),scope='Read identified Asimov rows only; no PDF evaluation or fit.'))


if __name__=='__main__':main()
