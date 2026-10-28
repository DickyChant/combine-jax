"""Verify the assigned GPU before any model loading or numerical benchmarks."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import sys
from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    devices = jax.devices()
    assert devices and all(d.platform == 'gpu' for d in devices) and jax.config.x64_enabled
    assert 'ROOT' not in sys.modules
    print('JAX GPU verified', [dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
          'JAX', jax.__version__, 'float64', jax.config.x64_enabled, flush=True)
    monitor = Resources(True, args.output)
    monitor.close()
    report = dict(provenance(), status='gpu_verified', script_sha256=sha(__file__), python=sys.version,
        versions={n: version(n) for n in ('jax','jaxlib','jax-cuda12-plugin','torch')}, float64=True,
        devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
        resources=monitor.snapshot(), root_imported=False, scope='Device verification only; no model benchmark.')
    save_json(args.output, report)
    print(report['devices'], report['versions'], flush=True)


if __name__ == '__main__':
    main()
