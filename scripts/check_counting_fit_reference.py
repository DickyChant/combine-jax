"""Check the standalone Combine fit reference against the tested JAX fit driver."""

import hashlib
import json
from pathlib import Path

import numpy as np

from hhvv_export_cache import load_export
from combine_jax.fitting import fit


def main():
    path = Path("reports/counting_combine_fit.json")
    reference = json.loads(path.read_text())
    assert reference["status"] == "reference_ready"
    artifact = Path(reference["arrays"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == reference["arrays_sha256"]
    likelihood = load_export("build/gpu_exports/counting.pkl",reference["source"])
    result = fit(likelihood,tolerance=1e-4)
    assert result.valid_minimum and result.covariance_status == "positive_definite"
    with np.load(artifact,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["parameter_names"],result.parameter_names)
        np.testing.assert_allclose(result.values,arrays["values"],atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(result.covariance,arrays["covariance"],atol=1e-4,rtol=1e-3)
        report = dict(status="passed",reference=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      max_parameter_absolute_error=float(abs(result.values-arrays["values"]).max()),
                      max_covariance_absolute_error=float(abs(result.covariance-arrays["covariance"]).max()),
                      nll_absolute_error=abs(result.nll-reference["canonical_fitted_nll"]))
        assert report["nll_absolute_error"] < 1e-6
    Path("reports/counting_combine_fit_check.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__ == "__main__":
    main()
