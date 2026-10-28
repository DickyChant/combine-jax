"""Summarize frozen native curvature samples without reading AD results."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from compare_gpu_curvature import first_converged_triple


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path("reports/gpu_curvature_saved_sample_diagnostic.json")
    if output.exists():
        raise FileExistsError(output)
    reference_path = Path("reports/gpu_combine_curvature_cached.json")
    pilot_path = Path("reports/gpu_combine_curvature_real_sum_pilot.json")
    reference = json.loads(reference_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    assert reference["status"] == "reference_uncertain"
    assert pilot["status"] == "pilot_checked"
    assert pilot["parent_reference_sha256"] == sha(reference_path)
    cases = []
    for case in reference["cases"]:
        assert case["selected_triple"] == first_converged_triple(case["probes"])
        slopes = []
        for probe in case["probes"]:
            h = probe["step"]
            samples = probe["samples"]
            assert len(samples) == 4
            assert [s["multiple"] for s in samples] == [-2, -1, 1, 2]
            delta = np.asarray([s["stable_delta"] for s in samples], dtype=np.longdouble)
            assert np.isfinite(delta).all()
            left = (delta[0] - 4*delta[1]) / (2*h)
            right = (4*delta[2] - delta[3]) / (2*h)
            slopes.append(dict(step=h, left_three_point_slope=float(left),
                               right_three_point_slope=float(right),
                               right_minus_left=float(right-left)))
        cases.append(dict(label=case["label"], selected_triple=case["selected_triple"],
                          curvature_estimates=[dict(step=p["step"], curvature=p["curvature"])
                                               for p in case["probes"]],
                          one_sided_slope_estimates=slopes))
    original = next(c for c in reference["cases"] if c["label"] == pilot["label"])
    assert original["direction"] == pilot["direction"]
    assert first_converged_triple(pilot["probes"]) == pilot["selected_triple"] is None
    corrections = []
    for before, after in zip(original["probes"], pilot["probes"], strict=True):
        assert before["step"] == after["step"]
        deltas = [abs(a["stable_delta"]-b["stable_delta"])
                  for a, b in zip(before["samples"], after["samples"], strict=True)]
        corrections.append(dict(step=before["step"], cached_curvature=before["curvature"],
                                real_sum_curvature=after["curvature"],
                                curvature_absolute_change=abs(before["curvature"]-after["curvature"]),
                                max_sample_delta_absolute_change=max(deltas)))
    unresolved = [c["label"] for c in cases if c["selected_triple"] is None]
    assert unresolved == reference["unresolved_directions"] and len(unresolved) == 10
    source_files = [
        Path(__file__), Path("scripts/compare_gpu_curvature.py"),
        Path("vendor/combine_pypi/vendor/CombinedLimit/src/CachingNLL.cc"),
        Path("vendor/combine_pypi/vendor/CombinedLimit/interface/CombineMathFuncs.h"),
        Path("src/combine_jax/translators.py"),
    ]
    report = dict(
        status="diagnostic_complete", created_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version, numpy_version=np.__version__,
        inputs={str(p): sha(p) for p in (reference_path, pilot_path)},
        source_sha256={str(p): sha(p) for p in source_files},
        scope="Saved native-only samples and source inspection. No ROOT or JAX evaluation, "
              "new reference point, AD array read, acceptance-policy change or performance measurement.",
        slope_formula="With stable f(0)=0: left=(f(-2h)-4f(-h))/(2h); "
                      "right=(4f(h)-f(2h))/(2h). These are finite-step diagnostics, "
                      "not proof of a limiting derivative or a cusp.",
        cases=cases, converged_directions=len(cases)-len(unresolved),
        unresolved_directions=unresolved,
        real_sum_pilot=dict(
            label=pilot["label"], selected_triple=pilot["selected_triple"],
            corrections=corrections,
            max_sample_delta_absolute_change=max(p["max_sample_delta_absolute_change"]
                                                 for p in corrections),
            interpretation="The real-sum stable-difference replacement does not produce "
                           "a converged triple for this pilot direction. Small changes in "
                           "NLL differences are amplified by inverse squared step size. "
                           "This does not identify the remaining source of uncertainty."),
        source_inspection=[
            dict(path=str(source_files[2]), symbol="ArgSetChecker::changed",
                 observation="Real values are compared with val != vals_[i]; category indices "
                             "are compared exactly. This checker has no numerical change tolerance.",
                 limitation="This does not characterize every ROOT cache path."),
            dict(path=str(source_files[3]), symbol="logKappaForX / asymPow",
                 observation="For abs(theta)<0.5 the implementation uses polynomial interpolation; "
                             "asymPow exponentiates theta times this interpolated log kappa.",
                 limitation="A simple piecewise-power cusp at zero cannot be assumed from the "
                            "class description. No new equivalence test is claimed here."),
        ],
        conclusion="Eight directions retain their original converged triples and ten remain "
                   "uncertain. These diagnostics do not authorize promotion of any unresolved reference.",
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(output, report["status"], "converged", report["converged_directions"],
          "uncertain", len(unresolved))
    print("pilot max stable-delta change", report["real_sum_pilot"]["max_sample_delta_absolute_change"])


if __name__ == "__main__":
    main()
