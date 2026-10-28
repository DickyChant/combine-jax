"""Accept an actual Combine fit timing only after its fitted numbers agree."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--tensor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    native = json.loads(args.native.read_text())
    measured = json.loads(args.tensor.read_text())
    assert native['status'] == 'measured_fit_pending_agreement'
    assert measured['status'] == 'measured_pending_validation' and measured['operation'] == 'fit'
    assert native['host'] == measured['host'], 'A same-node fit timing comparison requires the same host'
    assert sha(measured['arrays']) == measured['arrays_sha256']
    with np.load(measured['arrays'], allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    fit = native['fit']
    np.testing.assert_array_equal(fit['parameters'], data['parameter_names'])
    values = np.array(fit['values']); errors = np.array(fit['errors'])
    if 'covariance_arrays' in fit:
        assert sha(fit['covariance_arrays']) == fit['covariance_arrays_sha256']
        with np.load(fit['covariance_arrays'], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['parameter_names'], fit['parameters'])
            np.testing.assert_array_equal(arrays['values'], values)
            np.testing.assert_array_equal(arrays['errors'], errors)
            covariance = arrays['covariance']
    else:
        covariance = np.array(fit['covariance'])
    error = abs(fit['canonical_nll']-float(data['fit_value']))
    parameter_ratio = np.abs(values-data['fit_values'])/np.maximum(np.maximum(.01*np.abs(values), .01*errors), 1e-30)
    error_ratio = np.abs(errors-data['fit_reported_errors'])/np.maximum(errors, 1e-30)
    gates = dict(native_valid=fit['status'] == 0 and fit['covariance_quality'] == 3 and fit['roofit_evaluation_errors'] == 0,
        device_valid=bool(data['fit_status'] == 0 and data['fit_covariance_status'] == 0),
        nll=error < 1e-6, parameters=bool(np.all(parameter_ratio < 1)),
        uncertainties=bool(np.all(error_ratio < .01)),
        covariance=bool(np.allclose(data['fit_covariance'], covariance, rtol=.01, atol=1e-5)))
    report = dict(status='passed' if all(gates.values()) else 'failed', gates=gates, script_sha256=sha(__file__),
        native=str(args.native), native_sha256=sha(args.native), tensor=str(args.tensor), tensor_sha256=sha(args.tensor),
        host=native['host'], nll_absolute_error=error, max_parameter_error_over_tolerance=float(np.max(parameter_ratio)),
        max_reported_error_relative_difference=float(np.max(error_ratio)),
        native_executable_wall_seconds=native['executable_wall_seconds'],
        tensor_timings=measured['timings'],
        timing_scope='Native includes startup/input/MIGRAD/HESSE/output; tensor compile and synchronized fit execution are separate. A five-call warm median and a single executable run are not a statistical speedup study.')
    save_json(args.output, report)
    print(report['status'], gates, flush=True)
    assert report['status'] == 'passed'


if __name__ == '__main__':
    main()
