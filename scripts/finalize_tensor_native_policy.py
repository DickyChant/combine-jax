"""Correct only fixed-state metadata after the complete native-policy import."""
import argparse
import json
from pathlib import Path

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.tensor.fit_policy import workspace_fit_state,check_native_fit_state
from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.directory.exists() and not args.output.exists()
    source=Path('build/tensor_hhvv_native_policy/model')
    model=TensorBundle.load(source);info=model_parameter_info(model)
    from combine_jax.root_io import open_workspace,load_caching_nll
    from hhvv_setup import configure_hhvv
    load_caching_nll()
    with open_workspace('inputs/workspace_MX2000.root') as workspace:
        configure_hhvv(workspace)
        state=workspace_fit_state(workspace);previous=info.metadata['fit_policy']
        assert {k:v for k,v in state.items() if k!='fixed_real_values'}=={
            k:v for k,v in previous.items() if k!='fixed_real_values'}
        assert all(state['fixed_real_values'][n]==v for n,v in previous['fixed_real_values'].items())
        added={n:v for n,v in state['fixed_real_values'].items() if n not in previous['fixed_real_values']}
        expected={n:float(v) for n,v,lo,hi in zip(info.metadata['names'],info.arrays['initial'],
            info.arrays['lower'],info.arrays['upper'],strict=True) if lo==hi}
        assert added==expected and len(added)==4
        updated=TensorBundle({**model.metadata,'parameter_info':{**info.metadata,'fit_policy':state}},model.arrays)
        check_native_fit_state(workspace,updated)
    for name,a in model.arrays.items():
        b=updated.arrays[name];assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    args.directory.mkdir();updated.save(args.directory/'model')
    # Keep the accepted point/reference files; native-policy gates must be rerun.
    manifest=json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    case=manifest['hhvv'];old_model=case['model']
    case['model']=str(args.directory/'model')
    case['hashes']={p:h for p,h in case['hashes'].items() if not p.startswith(old_model+'/')}
    case['hashes'].update({str(p):sha(p) for p in (args.directory/'model').iterdir()})
    save_json(args.directory/'manifest.json',dict(hhvv=case))
    save_json(args.output,dict(status='passed_metadata_only_update',script_sha256=sha(__file__),
        source_hashes={str(source/n):sha(source/n) for n in ('model.json','arrays.npz')},
        added_fixed_values=added,unchanged_numeric_buffers=len(model.arrays),
        artifacts={str(p):sha(p) for p in [*(args.directory/'model').iterdir(),args.directory/'manifest.json']},
        policy_source_sha256=sha('src/combine_jax/tensor/fit_policy.py'),
        scope='Original full native-policy import is retained. Only four equal-bound parameter values are added to effective fixed-state metadata; every numeric buffer is unchanged. No new extraction, fit, or numerical agreement claim. Actual CLI preflight and native point gates remain required.'))
    print('passed',added,flush=True)


if __name__=='__main__':main()
