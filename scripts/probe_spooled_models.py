"""Validate complete local smoke-model Hessians with disk-backed executables."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import jax
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export
from spooled_executable import SpooledExecutable


def main():
    os.sched_setaffinity(0,[sorted(os.sched_getaffinity(0))[1]])
    assert jax.default_backend()=="cpu" and jax.config.jax_enable_x64 and "ROOT" not in sys.modules
    started=time.perf_counter()
    output=Path("reports/gpu_local_spooled_models.json")
    directory=Path("build/gpu_results/local_spooled_models")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior spooled-model evidence")
    directory.mkdir(parents=True)
    manifest_path=Path("build/gpu_exports/manifest.json");raw=manifest_path.read_bytes();manifest=json.loads(raw)
    report=dict(status="running",scope="Complete local CPU smoke-model Hessians; no GPU, full-HHVV or performance acceptance",
        jax_version=jax.__version__,jaxlib_version=__import__("jaxlib").__version__,float64=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)),backend="cpu",root_imported=False,
        manifest=str(manifest_path),manifest_sha256=hashlib.sha256(raw).hexdigest(),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path("scripts/spooled_executable.py"))},models=[])
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary=output.with_suffix(".json.tmp");temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");temporary.replace(output)
    try:
        for model in ("counting","automcstats"):
            config=manifest["models"][model]
            likelihood=load_export(Path("build/gpu_exports")/config["cache"],config["source"])
            evaluator=likelihood.local_evaluator()
            points=np.array([c["theta"] for c in config["reference"]["cases"][:3]])
            expected=np.array([evaluator.hessian(p) for p in points])
            case=dict(model=model,source=config["source"],points=len(points),parameters=len(likelihood.initial),
                      terms=len(evaluator.terms),serialized_terms=[])
            report["models"].append(case)
            for index,term in enumerate(evaluator.terms):
                compiled=term.hessian.lower(points[0,term.indices]).compile()
                stored,record=SpooledExecutable.write(compiled,directory/model/f"{index}.bin",backend="cpu")
                term.hessian=stored
                del compiled
                release_compilation_cache()
                case["serialized_terms"].append(dict(term=term.name,parameters=len(term.indices),**record))
            actual=np.array([evaluator.hessian(p) for p in points])
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual,expected)
            artifact=directory/(model+".npz")
            np.savez_compressed(artifact,parameter_names=likelihood.parameter_names,points=points,
                                original_hessians=expected,spooled_hessians=actual)
            case.update(status="passed",max_absolute_error=float(np.max(abs(actual-expected),initial=0.)),
                        arrays=str(artifact),arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        calls={t.name:t.hessian.calls for t in evaluator.terms})
            evaluator.clear_cache(release_cache=True)
            print(model,case["status"],"terms",case["terms"],"points",len(points),flush=True)
            save()
        report["status"]="passed"
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc());save();raise


if __name__=="__main__":
    main()
