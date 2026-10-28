"""Compiler allocation evidence for a complete tensor HHVV value+gradient."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-report',default='reports/tensor_hhvv_compact_import.json')
    parser.add_argument('--output',default='reports/tensor_hhvv_memory_baseline.json')
    args=parser.parse_args();path=Path(args.output);assert not path.exists()
    source=Path(args.import_report);imp=json.loads(source.read_bytes());assert imp['status']=='passed'
    for p,digest in imp['artifacts'].items():assert sha(p)==digest
    model=TensorBundle.load(imp['model']);arrays=jax.device_put(dict(model.arrays));initial=np.array(model.arrays['scalars__parameter_initial'])
    jax.config.update('jax_enable_compilation_cache',False)
    report=dict(status='running',script_sha256=sha(__file__),import_report=str(source),import_report_sha256=sha(source),
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))},
        jax_version=jax.__version__,hostname=platform.node(),backend=jax.default_backend(),float64=bool(jax.config.x64_enabled),
        cpu_affinity=sorted(os.sched_getaffinity(0)),xla_flags=os.environ.get('XLA_FLAGS'),
        scope='Compiler allocation analysis only. No numerical acceptance or warm-call benchmark.')
    assert report['float64'] and report['backend']=='cpu';path.write_text(json.dumps(report,indent=2)+'\n')
    function=jax.value_and_grad(make_model_function(model));started=time.perf_counter()
    lowered=jax.jit(function).lower(initial,arrays);report['lowering_seconds']=time.perf_counter()-started
    started=time.perf_counter();compiled=lowered.compile();report['compile_seconds']=time.perf_counter()-started
    memory=compiled.memory_analysis();fields=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')
    report.update(status='complete',compiler_memory={n:getattr(memory,n) for n in fields},
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    path.write_text(json.dumps(report,indent=2)+'\n');print(report['compiler_memory'],flush=True)


if __name__=='__main__':main()
