"""Reimport benchmark cards under the checked native fit policy, preserving inputs."""
import argparse
import json
from pathlib import Path
import time
import traceback

import jax
import numpy as np

from combine_jax.root_io import load_caching_nll,open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.fit_policy import POLICY,combine_v11_fit_context,check_native_fit_state
from combine_jax.tensor.model import from_workspace,make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.validation import validate_workspace
from tensor_benchmark_common import provenance,save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.directory.exists() and not args.output.exists()
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    original=Path('build/tensor_benchmark_inputs/manifest.json')
    baseline=json.loads(original.read_text())
    args.directory.mkdir()
    report=dict(provenance(),status='preparing',script_sha256=sha(__file__),
        source_manifest=str(original),source_manifest_sha256=sha(original),cases={},
        scope='Fresh native-policy models and 26-point native/old-AD/tensor gates. Copies small workspaces; does not change original benchmark inputs. Original benchmark initial values, explicit nuisance step metadata and discrete branches are written to each copy. No fit timing.')
    save_json(args.output,report);started=time.perf_counter()
    root=load_caching_nll()
    try:
        for label in ('counting','automcstats','parametric','multipdf_exp','multipdf_bern'):
            case=baseline[label]
            assert all(sha(p)==h for p,h in case['hashes'].items())
            old=TensorBundle.load(case['model']);info=model_parameter_info(old)
            directory=args.directory/label;directory.mkdir()
            card=directory/'card.txt';card.write_bytes((Path(case['workspace']).parent/'card.txt').read_bytes())
            # The card is used for membership/metadata during import; the saved
            # workspace already embeds its shapes, with no external file reads.
            copied=directory/'workspace.root'
            print('prepare',label,flush=True)
            with open_workspace(case['workspace']) as workspace:
                for name,c in case['configuration'].items():
                    workspace.var(name).setVal(c['value']);workspace.var(name).setConstant(c['constant'])
                for name,value,error in zip(info.metadata['names'],info.arrays['initial'],info.arrays['error'],strict=True):
                    workspace.var(name).setVal(float(value));workspace.var(name).setError(float(error))
                for c in old.metadata['provenance']['discrete_choices'].values():
                    workspace.cat(c['category']).setIndex(c['index'])
                workspace.writeToFile(str(copied))
                model=from_workspace(workspace,card,fit_policy=POLICY)
                check_native_fit_state(workspace,model)
                model.save(directory/'model')
                with combine_v11_fit_context(workspace):
                    native=validate_workspace(workspace,random_points=25,seed=1819)
            save_json(directory/'reference.json',native)
            assert native['max_nll_absolute_error']<1e-6
            assert native['max_gradient_relative_error']<1e-4
            assert native['max_hessian_scaled_error']<1e-6
            names=np.array(native['parameters']);points=np.array([r['theta'] for r in native['cases']],dtype=np.float64)
            np.testing.assert_array_equal(names,model.metadata['parameter_names'])
            arrays=jax.device_put(dict(model.arrays));function=make_model_function(model)
            vg=jax.jit(jax.value_and_grad(function));hessian=jax.jit(jax.hessian(function))
            captures=[jax.device_get(vg(p,arrays)) for p in points]
            values=np.array([v for v,g in captures]);gradients=np.array([g for v,g in captures])
            native_values=np.array([r['combine_nll'] for r in native['cases']])
            fd=np.array([r['combine_fd_gradient'] for r in native['cases']])
            old_grad=np.array([r['jax_gradient'] for r in native['cases']])
            matrices=np.array([jax.device_get(hessian(p,arrays)) for p in points[:3]])
            old_hessian=np.array([r['hessian'] for r in native['cases'][:3]])
            errors=dict(nll=float(np.max(abs(values-native_values))),
                native_gradient_relative=float(np.max(abs(gradients-fd)/np.maximum(abs(fd),1e-8))),
                old_ad_gradient_relative=float(np.max(abs(gradients-old_grad)/np.maximum(abs(old_grad),1e-8))),
                old_ad_hessian_absolute=float(np.max(abs(matrices-old_hessian))))
            assert errors['nll']<1e-6 and errors['native_gradient_relative']<1e-4 and errors['old_ad_gradient_relative']<1e-4,errors
            np.testing.assert_allclose(matrices,old_hessian,rtol=1e-8,atol=1e-8)
            np.savez_compressed(directory/'points.npz',points=points,parameter_names=names)
            np.savez_compressed(directory/'curvature_points.npz',points=points[:3],parameter_names=names)
            np.savez_compressed(directory/'reference.npz',points=points,parameter_names=names,values=native_values,
                gradients=old_grad,native_fd_gradients=fd,hessians=old_hessian)
            np.savez_compressed(directory/'tensor_capture.npz',points=points,values=values,gradients=gradients,hessians=matrices)
            report['cases'][label]=dict(workspace=str(copied),configuration={},model=str(directory/'model'),
                points=str(directory/'points.npz'),curvature_points=str(directory/'curvature_points.npz'),
                reference=str(directory/'reference.npz'),native_reference=str(directory/'reference.json'),
                parameters=len(names),points_count=len(points),native_gates=errors,
                hashes={str(p):sha(p) for p in directory.rglob('*') if p.is_file()})
            save_json(args.output,report);print(label,errors,flush=True)
            del arrays,vg,hessian,captures; jax.clear_caches()
        full_path=Path('build/tensor_hhvv_native_policy_final/manifest.json')
        full=json.loads(full_path.read_text())['hhvv']
        assert all(sha(p)==h for p,h in full['hashes'].items())
        gate_path=Path('reports/tensor_native_policy_checkpoint.json');gate=json.loads(gate_path.read_text())
        assert gate['status']=='passed_native_state_and_26_point_gates_fits_pending'
        assert all(sha(p)==h for p,h in gate['artifacts'].items())
        report['cases']['hhvv']=full
        manifest=args.directory/'manifest.json';save_json(manifest,report['cases'])
        report.update(status='prepared_and_validated',manifest=str(manifest),manifest_sha256=sha(manifest),
            hhvv_gate_sha256=sha(gate_path),root_version=root.gROOT.GetVersion())
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;save_json(args.output,report)


if __name__=='__main__':main()
