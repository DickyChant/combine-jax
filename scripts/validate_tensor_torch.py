"""Validate optional PyTorch execution of the exact portable benchmark artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['counting', 'automcstats'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('build/tensor_benchmark_inputs/manifest.json'))
    args = parser.parse_args()
    assert not args.output.exists()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import numpy as np
    import torch
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.torch_backend import make_torch_function, to_torch_arrays
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text())
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        torch_version=torch.__version__, float64=True, manifest_sha256=sha(manifest_path), cases=[],
        scope='CPU numerical acceptance using the same neutral buffers and frozen Combine/old-AD references. This is not a performance measurement or CUDA-graph acceptance.')
    save_json(args.output, report)
    monitor = Resources(False, args.output)
    try:
        for case in args.cases:
            source = manifest[case]
            assert all(sha(p) == digest for p, digest in source['hashes'].items())
            model = TensorBundle.load(source['model'])
            from tensor_benchmark_dataset import benchmark_dataset
            benchmark_dataset(source,model,label=case)
            fn = make_torch_function(model)
            arrays = to_torch_arrays(model)
            with np.load(source['reference'], allow_pickle=False) as a:
                reference = {k: a[k] for k in a.files}
            np.testing.assert_array_equal(reference['parameter_names'], model.metadata['parameter_names'])
            values, gradients, hessians = [], [], []
            for i, point in enumerate(reference['points']):
                theta = torch.tensor(point, dtype=torch.float64, requires_grad=True)
                value = fn(theta, arrays)
                gradient = torch.autograd.grad(value, theta)[0]
                values.append(value.detach().numpy())
                gradients.append(gradient.detach().numpy())
                if i < len(reference.get('hessians', [])):
                    hessians.append(torch.autograd.functional.hessian(lambda t: fn(t, arrays), theta).numpy())
                report.update(current_case=case, completed_points=i+1, resources=monitor.snapshot())
                save_json(args.output, report)
            actual = dict(values=np.asarray(values), gradients=np.asarray(gradients), hessians=np.asarray(hessians),
                          points=reference['points'], parameter_names=reference['parameter_names'])
            numeric = Path('build')/(args.output.stem+'_'+case+'.npz')
            assert not numeric.exists()
            np.savez_compressed(numeric, **actual)
            checks = {}
            for key, expected_key, tolerance in [('values','values',1e-6), ('gradients','gradients',1e-4),
                    ('gradients','native_fd_gradients',1e-4), ('hessians','hessians',1e-8)]:
                if expected_key not in reference:
                    continue
                difference = np.abs(actual[key]-reference[expected_key])
                relative = difference/np.maximum(np.abs(reference[expected_key]), 1e-8)
                passed = np.all(relative < tolerance) if key == 'gradients' else np.all(difference < tolerance)
                checks[expected_key] = dict(max_absolute_error=float(difference.max()),
                    max_relative_error=float(relative.max()), passed=bool(passed and np.isfinite(actual[key]).all()))
            row = dict(case=case, model=source['model'], model_sha256={n:sha(Path(source['model'])/n) for n in ('model.json','arrays.npz')},
                reference=source['reference'], reference_sha256=sha(source['reference']), arrays=str(numeric),
                arrays_sha256=sha(numeric), points=len(values), hessian_points=len(hessians), checks=checks,
                passed=all(c['passed'] for c in checks.values()),
                native_gradient_policy='separate full-HHVV native-policy audit required' if case == 'hhvv' else 'direct saved native FD gradients')
            report['cases'].append(row); save_json(args.output, report)
            print(case, row['passed'], checks, flush=True)
        assert not any(module in sys.modules for module in ('ROOT','jax'))
        assert all(sha(p) == digest for p, digest in report['sources'].items())
        report.update(status='passed' if all(c['passed'] for c in report['cases']) else 'failed',
                      root_imported=False, jax_imported=False)
        save_json(args.output, report)
        assert report['status'] == 'passed'
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        save_json(args.output, report)
        raise
    finally:
        monitor.close()
        report['resources'] = monitor.snapshot()
        save_json(args.output, report)


if __name__ == '__main__':
    main()
