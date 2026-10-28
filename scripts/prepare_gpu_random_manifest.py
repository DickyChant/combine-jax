"""Add the certified random NLL points to a separate, ROOT-free retry manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    base_path = Path("build/gpu_exports/manifest.json")
    parent_path = Path("reports/hhvv_random_values.json")
    base = json.loads(base_path.read_text())
    parent = json.loads(parent_path.read_text())
    model = base["models"]["hhvv"]
    assert parent["status"] == "passed" and parent["valid_random_points"] == 25
    assert parent["source"] == model["source"]
    assert parent["combine_version"] == base["combine_version"] and parent["root_version"] == base["root_version"]
    np.testing.assert_array_equal(model["reference"]["cases"][0]["theta"],parent["initial"])
    assert len(parent["cases"]) == 26
    assert all(c["nll_absolute_error"] < 1e-6 for c in parent["cases"])
    assert np.asarray([c["theta"] for c in parent["cases"]]).shape == (26,len(parent["parameter_names"]))
    model["reference"].update(
        parameter_names=parent["parameter_names"],
        cases=[dict(label="nominal" if c["attempt"] == 0 else f"random_attempt_{c['attempt']}",
                    theta=c["theta"],combine_nll=c["combine_nll"]) for c in parent["cases"]],
        random_nll_reference=str(parent_path),random_nll_reference_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        derivative_scope="Nominal derivatives checked against the frozen references; random gradient outputs retained for independent FD comparison, not accepted by this NLL-only manifest; nominal Hessian only")
    output = Path("build/gpu_exports/manifest_random.json")
    payload = (json.dumps(base,indent=2,allow_nan=False)+"\n").encode()
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError("Preserve the existing random-point manifest")
    output.write_bytes(payload)
    report = dict(status="prepared",source=parent["source"],base_manifest=str(base_path),
                  base_manifest_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                  random_nll_reference=str(parent_path),random_nll_reference_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                  manifest=str(output),manifest_sha256=hashlib.sha256(payload).hexdigest(),
                  hhvv_reference_points=26,parameters=len(parent["parameter_names"]),
                  scope="Prepared NLL references only; no new GPU/CPU execution or random-gradient acceptance")
    Path("reports/gpu_random_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__ == "__main__":
    main()
