"""Audit completed Asimov job 16969397.0; no fitting or resubmission."""
import json
from pathlib import Path
import numpy as np
from tensor_benchmark_common import sha, save_json


def main():
    raw=Path('build/asimov_a100_replacement_results')
    load=lambda p:json.loads(Path(p).read_text())
    native_path=Path('reports/gpu_tensor_asimov_a100_completed_native_fit.json')
    driver=load(raw/'gpu_tensor_edm_sequential_driver.json')
    fit=load(raw/'gpu_tensor_edm_sequential_fit.json')
    native=load(native_path)
    point=load('reports/gpu_tensor_asimov_a100_replacement_native_point.json')
    assert driver['native_reference_sha256']==sha(native_path)
    assert driver['dataset']=='asimov_prefit' and driver['same_host'] and driver['same_initial_point']
    assert point['dataset']==native['dataset']=='asimov_prefit'
    assert point['status']=='passed_point_only'
    assert point['fit_sha256']==sha('reports/gpu_tensor_asimov_a100_replacement_fit.json')
    assert native['host']==fit['host']=='b9g57n0024.cern.ch'
    archive=raw/'gpu_tensor_edm_sequential_fit.npz'
    assert sha(archive)==fit['arrays_sha256']==point['arrays_sha256']
    with np.load(archive) as a:
        names=a['parameter_names']; values=a['fit_values']; ref=np.array(native['fit']['values'])
        sigma=np.array(native['fit']['errors'])
        np.testing.assert_array_equal(names,native['fit']['parameters'])
        np.testing.assert_array_equal(values,point['point'])
        assert np.isfinite(values).all() and np.isfinite(sigma).all()
        delta=values-ref
        pulls=np.divide(abs(delta),sigma,out=np.zeros_like(sigma),where=sigma>0)
        order=np.argsort(pulls)[::-1]
        def item(i):
            return dict(name=str(names[i]),tensor=float(values[i]),native=float(ref[i]),
                native_migrad_error=float(sigma[i]),delta_over_sigma=float(pulls[i]))
        difference=float(a['fit_value'])-native['fit']['canonical_nll']
        minimum=bool(abs(difference)<=2e-4 and np.all(abs(delta)<=np.maximum(.01*abs(ref),.01*sigma)))
        summary=dict(final_nll=float(a['fit_value']),r_hhvv=item(list(names).index('r_hhvv')),
            corrected_estimated_edm=float(a['fit_corrected_estimated_edm']),
            iterations=int(a['fit_iteration']),optimizer_status=int(a['fit_status']),
            delta_nll=difference,max_parameter_pull=float(pulls.max()),top_parameters=[item(i) for i in order[:10]])
    operations={}
    for op in ('value_gradient','fit','covariance_row'):
        j=load(raw/f'gpu_tensor_edm_sequential_{op}.json'); m=j['compiler_memory']
        allocation=m['argument_size_in_bytes']+m['output_size_in_bytes']+m['temp_size_in_bytes']-m['alias_size_in_bytes']
        assert allocation<8*1024**3 and j['float64']
        operations[op]=dict(timings=j['timings'],static_bytes=allocation,
            resources=j['resources'],jax_allocator=j['jax_allocator'])
    row=load(raw/'gpu_tensor_edm_sequential_covariance_row.json')['covariance_result']
    report=dict(status='passed_same_minimum' if minimum else 'failed_same_minimum',
        script_sha256=sha(__file__),job='16969397.0',scheduler_status=4,worker_exit=load(raw/'worker_exit.json'),
        scheduler_wall_seconds=1372.,source_revision=driver['source_revision'],
        raw_artifacts={str(p):sha(p) for p in sorted(raw.iterdir()) if p.is_file()},
        native_reference=str(native_path),native_reference_sha256=sha(native_path),
        dataset=driver['dataset'],dataset_identity=driver['dataset_identity'],
        same_host=True,same_initial_point=True,environment=fit['environment'],
        jax_version=fit['jax_version'],jaxlib_version=fit['jaxlib_version'],devices=fit['devices'],
        native_same_point_errors=point['errors'],fit=summary,
        native_migrad_seconds=native['migrad_seconds'],
        tensor_evaluations=fit['evaluation_counts'],native_evaluations=native['evaluation_counts'],
        operations=operations,covariance_row=row,
        accepted_speed_ratio=None,uncertainty_comparison=None,
        scope='Asimov only. Estimated EDM convergence is not minimum agreement. No accepted speed ratio on a failed minimum gate. CG did not solve the row; native saved errors are Migrad errors, not an explicit Hesse reference. Device-wide memory includes baseline/other users; process attribution was unavailable.')
    save_json(Path('reports/gpu_tensor_asimov_a100_replacement_collection.json'),report)
    print(report['status'],summary['delta_nll'],summary['max_parameter_pull'])


if __name__=='__main__':main()
