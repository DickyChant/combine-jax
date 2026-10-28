"""Compare each parser result directly with its own saved RooFitResult entries."""
import json
from pathlib import Path

import numpy as np

from combine_jax.root_io import load_root
from tensor_benchmark_common import save_json, sha


def main():
    output = Path('reports/gpu_tensor_native_fit_parser_audit.json')
    assert not output.exists()
    root = load_root()
    rows = []
    for version in (3,4):
        for case in ('counting','automcstats'):
            path = Path(f'reports/gpu_tensor_local_{case}_combine_fit_parser_v{version}.json')
            report = json.loads(path.read_text())
            assert report['status'] == 'measured_fit_pending_agreement'
            result_path = Path(report['cwd'])/'multidimfit.tensor_benchmark.root'
            assert sha(result_path) == report['result_sha256']
            handle = root.TFile.Open(str(result_path))
            fit = handle.Get('fit_mdf')
            parameters = fit.floatParsFinal()
            names = [p.GetName() for p in parameters]
            selected = [names.index(n) for n in report['fit']['parameters']]
            matrix = fit.covarianceMatrix()
            expected = np.array([[float(matrix[i][j]) for j in selected] for i in selected])
            actual = np.array(report['fit']['covariance'])
            difference = np.abs(actual-expected)
            equal = bool(np.array_equal(actual, expected))
            assert equal == (version == 4)
            np.testing.assert_array_equal(report['fit']['values'], [parameters[i].getVal() for i in selected])
            np.testing.assert_array_equal(report['fit']['errors'], [parameters[i].getError() for i in selected])
            assert int(fit.status()) == 0 and int(fit.covQual()) == 3
            rows.append(dict(case=case, parser_version=version, report=str(path), report_sha256=sha(path),
                result_sha256=sha(result_path), covariance_bitwise_equal=equal,
                covariance_max_absolute_error=float(difference.max()),
                fitted_values_and_reported_errors_bitwise_equal=True))
            handle.Close()
    save_json(output, dict(status='passed_corrected_parser_with_failed_v3_preserved',
        script_sha256=sha(__file__), parser_sha256=sha('scripts/benchmark_combine_fit.py'), cases=rows,
        scope='The corrected parser copies every small-card native matrix entry exactly, including the marginal autoMCStats submatrix. This does not resolve or relax native-versus-AD covariance discrepancies. Full HHVV native execution remains pending.'))
    print('passed corrected parser; prior failed covariance copies retained', flush=True)


if __name__ == '__main__':
    main()
