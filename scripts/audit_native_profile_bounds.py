"""Check saved native CLI profiled-nuisance bounds without another fit."""
import argparse
import json
from pathlib import Path

from native_fit_preflight import check_profile_bounds
from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    report=json.loads(args.preflight.read_text());assert report['status']=='passed_preflight_only'
    outputs=list((Path(report['cwd'])/'preflight').glob('higgsCombine*.root'))
    assert len(outputs)==1 and sha(outputs[0])==report['preflight']['workspace_sha256']
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.root_io import open_workspace
    model=TensorBundle.load(args.model)
    with open_workspace(outputs[0]) as workspace:
        rows=check_profile_bounds(workspace.getSnapshot('MultiDimFit'),model)
    save_json(args.output,dict(status='passed_native_profile_bounds',script_sha256=sha(__file__),
        helper_sha256=sha('scripts/native_fit_preflight.py'),preflight_sha256=sha(args.preflight),
        workspace_sha256=sha(outputs[0]),model_hashes={str(args.model/n):sha(args.model/n) for n in ('model.json','arrays.npz')},
        checked=len(rows),profiled_bounds=rows,scope='Actual CLI snapshot limits for the analytically profiled nuisances; no new minimization or covariance claim.'))
    print('passed',len(rows),'native profiled nuisance bounds',flush=True)


if __name__=='__main__':main()
