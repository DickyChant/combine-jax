"""Record effective XLA settings for options before/after a legacy bare token."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,available[1:])
    code = '''
import json
import jax
from jaxlib import xla_client
devices = jax.devices()
# Keep every owner alive: debug_options is a reference to a nested C++ object.
options = xla_client.CompileOptions()
build = options.executable_build_options
debug = build.debug_options
print(json.dumps(dict(jax_version=jax.__version__,devices=[str(d) for d in devices],
                     backend_optimization_level=debug.xla_backend_optimization_level,
                     disabled_hlo_passes=debug.xla_disable_hlo_passes)))
'''
    variants = {
        "after_legacy": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 --xla_backend_optimization_level=0 --xla_disable_hlo_passes=constant_folding",
        "before_legacy": "--xla_cpu_multi_thread_eigen=false --xla_backend_optimization_level=0 --xla_disable_hlo_passes=constant_folding intra_op_parallelism_threads=1",
        "without_legacy": "--xla_cpu_multi_thread_eigen=false --xla_backend_optimization_level=0 --xla_disable_hlo_passes=constant_folding",
    }
    report = dict(status="running",python=sys.version,cpu_affinity=sorted(os.sched_getaffinity(0)),
                  scope="Local effective-option probe; no likelihood or performance measurement",cases=[])
    for label,flags in variants.items():
        process = subprocess.run([sys.executable,"-c",code],env={**os.environ,"XLA_FLAGS":flags,"JAX_PLATFORMS":"cpu"},
                                 capture_output=True,text=True,check=True,timeout=60)
        effective = json.loads(process.stdout)
        assert effective["backend_optimization_level"] == (3 if label == "after_legacy" else 0)
        assert effective["disabled_hlo_passes"] == ("" if label == "after_legacy" else "constant_folding")
        report["cases"].append(dict(label=label,xla_flags=flags,effective=effective,stderr=process.stderr))
        print(label,effective,flush=True)
    report.update(status="confirmed",finding="Options after the bare legacy token were ignored; options before it or without it applied.")
    Path("reports/hhvv_xla_flag_order.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
