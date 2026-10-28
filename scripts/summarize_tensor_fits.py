"""Package logged native fit gates, rechecking numerical tolerances and sources."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    log=Path('reports/tensor_fitting_tests.log');output=Path('reports/tensor_device_fits.json')
    assert not output.exists()
    text=log.read_text();assert '20 passed, 2 warnings' in text and 'FAILED' not in text
    cases=[json.loads(s) for s in re.findall(r'FIT_VALIDATION_JSON=(\{[^\n]*\})',text)]
    assert [c['kind'] for c in cases]==['counting','automcstats']
    for c in cases:
        values=np.array(c['device_values']);native=np.array(c['native_values'])
        errors=np.array(c['device_errors']);native_errors=np.array(c['native_errors'])
        assert np.isfinite(values).all() and np.isfinite(errors).all() and np.all(errors>0)
        tolerance=np.maximum(.01*np.abs(native),.01*native_errors)
        ratio=float(np.max(np.abs(values-native)/tolerance))
        assert ratio==c['max_parameter_error_over_tolerance']<1.
        error_ratio=float(np.max(np.abs(errors-native_errors)/native_errors))
        assert error_ratio==c['max_error_relative_difference']<.01
        assert abs(c['device_nll']-c['native_nll'])==c['nll_absolute_error']<1e-6
        covariance=np.array(c['device_covariance']);reference=np.array(c['native_covariance'])
        np.testing.assert_allclose(np.sqrt(np.diag(covariance)),errors,atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(covariance,reference,atol=1e-5,rtol=.01)
        assert c['covariance_status']==0 and c['covariance_inverse_residual']<1e-6
        assert c['scaled_projected_gradient_max']<=1e-7 and c['old_hessian_max_absolute_error']<1e-9
        assert len(c['profiles'])==2
        for p in c['profiles']:
            assert p['status']==0 and p['delta_nll']>0
            assert abs(p['device_nll']-p['native_nll'])==p['nll_absolute_error']<1e-6
    baseline='dca96c2';old_sources={}
    for path in sorted(Path('src/combine_jax').glob('*.py')):
        previous=subprocess.run(['git','show',f'{baseline}:{path}'],capture_output=True,check=True).stdout
        assert hashlib.sha256(previous).hexdigest()==sha(path)
        old_sources[str(path)]=sha(path)
    files=list(Path('src/combine_jax/tensor').glob('*.py'))+[Path('tests')/n for n in (
        'test_tensor_parameter_info.py','test_tensor_optimize.py','test_tensor_fitting.py','conftest.py')]
    files += [Path('scripts/build_template_fixture.py'),Path('tests/cards/counting.txt')]
    report=dict(status='passed',script_sha256=sha(__file__),test_log=str(log),test_log_sha256=sha(log),tests=20,
        sources={str(p):sha(p) for p in sorted(files)},old_backend_unchanged_since=baseline,old_backend_sources=old_sources,
        packages={n:version(n) for n in ('higgs-combine-tool','ROOT','jax','jaxlib','iminuit','numpy')},
        scope='Local CPU float64 numerical validation. Native Combine v11 CachingSimNLL is minimized with iminuit; no separate Combine executable fit timing or GPU performance acceptance.',
        optimizer='Scaled box-constrained BFGS with device iteration and Armijo loops; all coordinates and model arrays are runtime inputs.',
        covariance='Raw inverse of the exact AD free Hessian, converted to physical units. No pseudoinverse, positive-definite repair or BFGS covariance.',
        gates=dict(nll_absolute=1e-6,parameter='max(1% of absolute native value, 0.01 native sigma)',
            error_relative=.01,profile_nll_absolute=1e-6,scaled_projected_gradient=1e-7),cases=cases,
        limitations=['Native fit gate currently covers counting and autoMCStats.',
            'Workspace importer attachment is pending; tests attach metadata from the retained extractor and serialize it before fitting.',
            'No accepted HHVV minimum, uncertainty or profile result.',
            'Validation overlaps the full HHVV Hessian run; no fit performance claims.'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('passed',[(c['kind'],c['nll_absolute_error'],c['max_error_relative_difference']) for c in cases],flush=True)


if __name__=='__main__':main()
