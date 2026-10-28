"""Blinded full-HHVV GPU expected limit on an explicitly prepared v10 artifact."""
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
    os.environ.update(JAX_PLATFORMS='cuda', JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu'
    assert any(f in devices[0].device_kind for f in ('H100', 'A100'))
    assert jax.config.x64_enabled and 'ROOT' not in sys.modules
    print('GPU verified', devices[0].device_kind, jax.__version__, flush=True)
    manifest_path = Path('build/hhvv_limit_runtime.json')
    manifest = json.loads(manifest_path.read_text())
    mass = manifest['mass']
    output = Path('reports')/('gpu_hhvv_MX'+str(mass)+'_expected_limit.json')
    assert not output.exists()
    report = dict(provenance(), status='running', mass=mass, script_sha256=sha(__file__),
        source_revision=manifest['source_revision'], manifest_sha256=sha(manifest_path),
        devices=[dict(kind=d.device_kind, platform=d.platform) for d in devices],
        versions={k: version(k) for k in ('jax', 'jaxlib', 'jax-cuda12-plugin', 'numpy', 'scipy')},
        float64=True, static_guard_gib=8., template_convention='v10_density',
        discrete_policy=manifest['discrete_policy'], profiles=[], jax_allocator_samples=[],
        scope='Blinded pre-fit background Asimov; v10 density convention, float64, unchanged 8 GiB static guard. Fixed RooMultiPdf choices, when requested, are explicit and do not constitute discrete profiling. Native quantile comparison pending reference.')
    save_json(output, report)
    monitor = Resources(True, output)
    original = None
    try:
        for name, checksum in manifest['files'].items():
            assert sha(name) == checksum, name
        prep = json.loads(Path('build/hhvv_limit/preparation.json').read_text())
        assert prep['status'] == 'imported_native_v10_background_asimov'
        assert prep['generation']['mass'] == mass and prep['generation']['signal_strength'] == 0.
        assert prep['generation']['combine_version'] == '10.3.3'
        report['preparation'] = prep
        from combine_jax.tensor.bundle import TensorBundle
        from combine_jax.methods.runner import run_expected_model
        from combine_jax.tensor.limit_profile import LimitProfiler
        start = time.perf_counter()
        model = TensorBundle.load('build/hhvv_limit/model')
        report['artifact_load_seconds'] = time.perf_counter()-start
        assert model.metadata['provenance']['template_convention'] == 'v10_density'
        assert model.metadata['asymptotic_asimov']['mass'] == mass
        assert model.metadata['asymptotic_asimov']['native_generation'] is True
        original = LimitProfiler._accept
        def progress(self, result, kind, mu):
            try:
                return original(self, result, kind, mu)
            finally:
                report['profiles'] = list(self.records)
                report['timings'] = dict(self.timings)
                report['compiler_memory'] = self.compiler_memory
                report['jax_allocator_samples'].append(dict(stage=kind, mu=mu, stats=devices[0].memory_stats()))
                report['resources'] = monitor.snapshot()
                monitor.stage = 'profiles'
                save_json(output, report)
                print('profile', kind, mu, self.records[-1], flush=True)
        LimitProfiler._accept = progress
        monitor.stage = 'compile'
        start = time.perf_counter()
        result = run_expected_model(model, 'r_hhvv', compiler_memory_limit_gib=8.,
                                    discrete_policy=manifest['discrete_policy'])
        report.update(result, prepared_model_wall_seconds=time.perf_counter()-start)
        assert len(result['limits']) == 5
        report['all_fits_converged'] = bool(result['profile_fits']) and all(r['status'] == 0 for r in result['profile_fits'])
        assert report['all_fits_converged']
        report['native_quantile_validation'] = 'pending_reference_and_matching_discrete_configuration'
    except BaseException as exc:
        report.update(status='failed_execution', error=repr(exc)); raise
    finally:
        if original is not None:
            LimitProfiler._accept = original
        monitor.close()
        report.update(resources=monitor.snapshot(), jax_allocator_final=devices[0].memory_stats(),
            process_seconds=time.perf_counter()-started, root_imported='ROOT' in sys.modules)
        save_json(output, report)
    print(report['status'], mass, report['limits'], flush=True)


if __name__ == '__main__':
    main()
