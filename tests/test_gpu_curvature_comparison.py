"""Exercise report acceptance using independent quartic finite differences."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize("outcome", ("passed", "failed", "reference_uncertain", "hash_error",
                                     "extension_hash_error", "changed_selection"))
def test_curvature_report_acceptance(tmp_path, outcome):
    parent = tmp_path/"parent.json"
    parent.write_text("{}\n")
    reference = dict(status="reference_ready", source={"synthetic_polynomial": True},
                     parent_reference=str(parent), parent_reference_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                     parameter_names=["x", "y"], initial=[0.,0.], policy="Independent quartic reference", cases=[])
    steps = (1e-3,5e-4,2e-4)
    for i, direction in enumerate(np.random.default_rng(721).normal(size=(18,2))):
        probes = []
        for step in steps:
            points = np.array([-2,-1,1,2],dtype=np.longdouble)[:,None]*np.longdouble(step)*direction
            x,y = points.T
            values = 7*x+11*y+x*x+3*x*y+5*y*y+x**4+y**4
            curvature = float(np.dot(values,[-1,16,16,-1])/(12*np.longdouble(step)**2))
            probes.append(dict(step=step,curvature=curvature))
        changes = [abs(a["curvature"]-b["curvature"])/abs(b["curvature"]) for a,b in zip(probes,probes[1:])]
        assert max(changes) < 1e-9
        reference["cases"].append(dict(label=str(i),direction=direction.tolist(),probes=probes,
                                       selected_triple=dict(first_probe_index=0,steps=steps,adjacent_relative_changes=changes)))
    if outcome == "reference_uncertain":
        reference["status"] = "reference_uncertain"
        reference["cases"][0]["selected_triple"] = None
        for probe in reference["cases"][0]["probes"]:
            # The |s| cusp has no finite second derivative at s=0.
            step = np.longdouble(probe["step"])
            values = np.abs(np.array([-2,-1,1,2])*step)
            probe["curvature"] = float(np.dot(values,[-1,16,16,-1])/(12*step**2))
    if outcome == "extension_hash_error":
        previous = tmp_path/"previous.json"
        previous.write_text("{}\n")
        reference.update(extends_reference=str(previous),extends_reference_sha256="incorrect")
    if outcome == "changed_selection":
        case = reference["cases"][0]
        case["probes"].append(dict(case["probes"][-1],step=1e-4))
        chosen = case["probes"][1:]
        case["selected_triple"] = dict(first_probe_index=1,steps=[p["step"] for p in chosen],
            adjacent_relative_changes=[abs(a["curvature"]-b["curvature"])/abs(b["curvature"])
                                       for a,b in zip(chosen,chosen[1:])])
    reference_path = tmp_path/"reference.json"
    reference_path.write_text(json.dumps(reference))
    matrix = np.array([[2.,3.],[3.,10.]])
    if outcome == "failed":
        matrix *= 1.01
    artifact = tmp_path/"arrays.npz"
    np.savez(artifact,parameter_names=reference["parameter_names"],point=reference["initial"],hessian=matrix)
    benchmark = dict(source=reference["source"],arrays=str(artifact),arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    if outcome == "hash_error":
        benchmark["arrays_sha256"] = "incorrect"
    benchmark_path = tmp_path/"benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark))
    script = Path(__file__).resolve().parents[1]/"scripts/compare_gpu_curvature.py"
    result = subprocess.run([sys.executable,str(script),str(benchmark_path),"--reference",str(reference_path)],capture_output=True,text=True)
    assert result.returncode == (0 if outcome == "passed" else 1), result.stderr
    report_path = tmp_path/"benchmark_combine_curvature.json"
    if outcome in ("hash_error","extension_hash_error","changed_selection"):
        assert not report_path.exists()
    else:
        report = json.loads(report_path.read_text())
        assert report["status"] == outcome
        if outcome == "failed":
            assert len(report["failed_directions"]) == 18
        if outcome == "reference_uncertain":
            assert report["unresolved_directions"] == ["0"]
