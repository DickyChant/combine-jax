"""Derive short-stencil references from saved stable ROOT differences, without AD."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fd_reference import estimates, first_consistent_triple


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(str(arguments.output))
    raw = arguments.reference.read_bytes()
    reference = json.loads(raw)
    assert reference["status"] in ("reference_ready", "reference_uncertain")
    assert hashlib.sha256(Path(reference["selection"]).read_bytes()).hexdigest() == reference["selection_sha256"]
    result = dict(source=reference["source"], reference=str(arguments.reference),
                  reference_sha256=hashlib.sha256(raw).hexdigest(),
                  script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path("scripts/fd_reference.py"))},
                  combine_version=reference["combine_version"], root_version=reference["root_version"],
                  policy="At each unchanged step, use second-order central/forward/backward and fourth-order central estimates, all within +/-2h. Select the first three consecutive steps whose twelve estimates have full relative spread below 1e-5. Later AD must agree with all twelve at relative tolerance 1e-4. No AD arrays or distances are read.",
                  scope="Alternative reference estimator after directional precision limits; all original probes and failed/uncertain comparisons remain unchanged. No new ROOT calls and no AD acceptance.",
                  points=[])
    for original in reference["points"]:
        assert original["canonical_parent_absolute_error"] < 1e-6
        assert original["canonical_restoration_absolute_error"] < 1e-6
        point = {key:original[key] for key in ("point_number","attempt","name","index","theta","coordinate","scale")}
        point["probes"] = []
        for probe in original["probes"]:
            checks = probe["delta_checks"]
            assert [d["multiple"] for d in checks] == [-4,-3,-2,-1,1,2,3,4]
            for check in checks:
                assert check["coordinate"] == original["coordinate"]+check["multiple"]*probe["step"]
                assert check["roofit_evaluation_errors"] == 0
                assert abs(check["stable_delta"]-check["combine_delta"]) < 1e-6
            deltas = {d["multiple"]:d["stable_delta"] for d in checks}
            values = estimates(deltas,probe["step"])
            assert abs(values["central_four"]-probe["stable"]["central"])/max(abs(probe["stable"]["central"]),1e-8) < 1e-9
            point["probes"].append(dict(step=probe["step"],fraction=probe["fraction"],estimates=values))
        point["selected_triple"] = first_consistent_triple(point["probes"])
        point["status"] = "reference_ready" if point["selected_triple"] is not None else "reference_uncertain"
        result["points"].append(point)
    result["status"] = "reference_ready" if all(p["status"] == "reference_ready" for p in result["points"]) else "reference_uncertain"
    arguments.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(arguments.output,result["status"],[(p["point_number"],p["status"],p["selected_triple"]) for p in result["points"]])


if __name__ == "__main__":
    main()
