"""GPU execution of the frozen, v10-validated noVH MX4000 expected-limit artifact."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import time

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    started = time.perf_counter()
    assert socket.gethostname().split('.')[0] != 'lxplus905'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    assert any(family in devices[0].device_kind for family in ('H100', 'A100'))
    assert jax.config.x64_enabled and 'ROOT' not in sys.modules
    print('GPU verified', devices[0].device_kind, jax.__version__, flush=True)
    output = Path('reports/gpu_vv4q_noVH_MX4000_density_limit.json')
    assert not output.exists()
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        devices=[dict(kind=d.device_kind, platform=d.platform) for d in devices],
        versions={k: version(k) for k in ('jax', 'jaxlib', 'jax-cuda12-plugin', 'numpy', 'scipy')},
        float64=True, static_guard_gib=8., template_convention='v10_density',
        quantile_relative_tolerance=1e-6, profiles=[], jax_allocator_samples=[],
        scope='One authorized GPU expected-limit run, exact CPU-validated neutral artifact; no ROOT loaded. All fits must converge and all five quantiles must match native v10 within 1e-6 relative.')
    save_json(output, report)
    monitor = Resources(True, output)
    original = None
    try:
        manifest_path = Path('build/gpu_density_limit_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        for name, checksum in manifest['files'].items():
            assert sha(name) == checksum, name
        base = Path('build/asymptotic_vv4q_noVH_MX4000_density')
        prep = json.loads((base / 'preparation.json').read_text())
        reference_path = Path('reports/asymptotic_vv4q_noVH_MX4000_density_limit_comparison.json')
        reference = json.loads(reference_path.read_text())
        gate = json.loads(Path('reports/asymptotic_vv4q_noVH_MX4000_density_gate.json').read_text())
        assert gate['status'] == 'passed' and reference['status'] == 'passed'
        for name, checksum in prep['model_sha256'].items():
            assert sha(base / 'model' / name) == checksum
        report.update(manifest_sha256=sha(manifest_path), source_revision=manifest['source_revision'],
            reference_sha256=sha(reference_path), preparation=prep,
            cpu_reference_seconds=reference['total_method_seconds'], native_v10_seconds=reference['native_seconds'])
        from combine_jax.tensor.bundle import TensorBundle
        from combine_jax.methods.runner import run_expected_model
        from combine_jax.tensor.limit_profile import LimitProfiler
        setup_start = time.perf_counter()
        model = TensorBundle.load(base / 'model')
        assert model.metadata['provenance']['template_convention'] == 'v10_density'
        report['artifact_load_seconds'] = time.perf_counter() - setup_start
        original = LimitProfiler._accept

        def progress(self, result, kind, mu):
            value = original(self, result, kind, mu)
            report['profiles'].append(dict(self.records[-1]))
            report['timings'] = dict(self.timings)
            report['compiler_memory'] = self.compiler_memory
            report['jax_allocator_samples'].append(dict(stage=kind, mu=mu, stats=devices[0].memory_stats()))
            report['resources'] = monitor.snapshot()
            monitor.stage = 'profiles'
            save_json(output, report)
            print('profile', kind, mu, self.records[-1], flush=True)
            return value

        LimitProfiler._accept = progress
        monitor.stage = 'compile'
        start = time.perf_counter()
        result = run_expected_model(model, 'r_hhvv', compiler_memory_limit_gib=8.)
        report.update(result, prepared_model_wall_seconds=time.perf_counter() - start)
        expected = [row['native_v10'] for row in reference['rows']]
        errors = [abs(a-b)/abs(b) for a, b in zip(result['limits'], expected, strict=True)]
        all_converged = bool(result['profile_fits']) and all(r['status'] == 0 for r in result['profile_fits'])
        passed = all_converged and max(errors) <= 1e-6
        report.update(status='passed' if passed else 'failed_native_v10_gate',
            native_v10_limits=expected, relative_errors=errors, all_fits_converged=all_converged,
            maximum_relative_error=max(errors),
            total_method_seconds_with_shared_preparation=prep['preparation_seconds'] + report['artifact_load_seconds'] + report['prepared_model_wall_seconds'],
            timing_scope='Preparation is the original measured CPU import, reused exactly; GPU artifact load, setup, lowering, compilation and execution are measured on this worker. CPU/native comparisons span different hosts, not a same-node speed certification.')
        if not passed:
            raise RuntimeError('GPU limit failed the unchanged convergence or native-v10 quantile gate')
    except BaseException as exc:
        if report['status'] == 'running':
            report['status'] = 'failed_execution'
        report['error'] = repr(exc)
        raise
    finally:
        if original is not None:
            LimitProfiler._accept = original
        monitor.close()
        report.update(resources=monitor.snapshot(), jax_allocator_final=devices[0].memory_stats(),
                      process_seconds=time.perf_counter()-started, root_imported='ROOT' in sys.modules)
        save_json(output, report)
    print(report['status'], report['limits'], report['relative_errors'], flush=True)


if __name__ == '__main__':
    main()
