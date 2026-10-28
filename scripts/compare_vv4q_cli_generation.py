"""Cross v10 CLI, v11 CLI and v11 direct Asimov generation."""
from pathlib import Path
import argparse
import numpy as np
from tensor_benchmark_common import sha, save_json
from compare_vv4q_asimov_datasets import bindings
from combine_jax.root_io import load_root, open_workspace
from combine_jax.dataset_identity import workspace_dataset_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background-only',action='store_true')
    args=parser.parse_args()
    root=load_root()
    report=dict(status='comparing',script_sha256=sha(__file__),datasets={},comparisons={},
        cli_generation_r_hhvv=0. if args.background_only else 1.,
        background_generation_control_eligible=args.background_only)
    data={}
    with open_workspace('build/asymptotic_vv4q_noVH_MX4000/workspace.root') as ws:
        direct=ws.data('combine_jax_asimov_bonly')
        data['v11_direct']=bindings(ws,direct)
        report['datasets']['v11_direct']=dict(identity=workspace_dataset_identity(ws,direct))
        sources=[('v10_cli','build/vv4q_v10_asimov_bonly'),('v11_cli','build/vv4q_v11_cli_asimov_bonly')] if args.background_only else [('v10_cli','build/vv4q_v10_asimov_verified'),('v11_cli','build/vv4q_v11_cli_asimov')]
        for name,directory in sources:
            path,=Path(directory).glob('higgsCombine*.root')
            handle=root.TFile.Open(str(path),'READ')
            generated=handle.Get('toys/toy_asimov')
            data[name]=bindings(ws,generated)
            report['datasets'][name]=dict(path=str(path),sha256=sha(path),identity=workspace_dataset_identity(ws,generated))
            handle.Close()
    for left,right in [('v10_cli','v11_cli'),('v11_cli','v11_direct')]:
        result={}
        for channel,a in data[left].items():
            b=data[right][channel]
            err=abs(a.weights-b.weights)
            result[channel]=dict(coordinates_equal=all(np.array_equal(a.coordinates[n],b.coordinates[n]) for n in a.coordinates),
                max_weight_absolute_error=float(max(err)),weight_absolute_error_sum=float(sum(err)))
        report['comparisons'][left+'_vs_'+right]=dict(identical=report['datasets'][left]['identity']==report['datasets'][right]['identity'],channels=result)
    report['status']='completed'
    output=Path('reports/asymptotic_vv4q_noVH_MX4000_cli'+('_bonly' if args.background_only else '')+'_generation_comparison.json')
    assert not output.exists()
    save_json(output,report)
    print(report['comparisons'],flush=True)


if __name__=='__main__':main()
