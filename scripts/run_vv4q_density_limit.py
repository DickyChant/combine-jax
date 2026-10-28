"""User-authorized corrected noVH MX4000 expected-limit comparison against dhi v10."""
import json
from pathlib import Path
import time
START=time.perf_counter()

from tensor_benchmark_common import Resources,provenance,sha,save_json


def main():
    output=Path('reports/asymptotic_vv4q_noVH_MX4000_density_limit.json')
    assert not output.exists()
    gate_path=Path('reports/asymptotic_vv4q_noVH_MX4000_density_gate.json')
    gate=json.loads(gate_path.read_text());assert gate['status']=='passed'
    base=Path('build/asymptotic_vv4q_noVH_MX4000_density')
    previous=json.loads(Path('reports/asymptotic_vv4q_noVH_MX4000.json').read_text())
    prep=json.loads((base/'preparation.json').read_text())
    for name,checksum in prep['model_sha256'].items(): assert sha(base/'model'/name)==checksum
    report=dict(provenance(),status='running',script_sha256=sha(__file__),gate_sha256=sha(gate_path),
        preparation=prep,template_convention='v10_density',reference_limits=previous['reference_limits'],
        reference_quantiles=previous['reference_quantiles'],reference_sha256=previous['reference_sha256'],
        native_wall_seconds=21957.58,profiles=[],
        scope='Authorized noVH MX4000 nominal background pre-fit Asimov expected limits. v10 reference selected explicitly for variable bins. Different hosts/core allocation; timing ratio is not a same-node speed claim.')
    save_json(output,report);monitor=Resources(False,output)
    try:
        from combine_jax.tensor.bundle import TensorBundle
        from combine_jax.methods.runner import run_expected_model
        from combine_jax.tensor.limit_profile import LimitProfiler
        model=TensorBundle.load(base/'model')
        assert model.metadata['provenance']['template_convention']=='v10_density'
        original=LimitProfiler._accept
        def progress(self,result,kind,mu):
            value=original(self,result,kind,mu)
            report['profiles'].append(dict(self.records[-1]));save_json(output,report)
            print('profile',kind,mu,'iterations',self.records[-1]['iteration'],flush=True)
            return value
        LimitProfiler._accept=progress
        then=time.perf_counter();result=run_expected_model(model,'r_hhvv')
        report.update(result,prepared_model_wall_seconds=time.perf_counter()-then)
        errors=[abs(a-b)/abs(b) for a,b in zip(result['limits'],report['reference_limits'],strict=True)]
        report.update(status='passed_quantiles' if max(errors)<.01 else 'failed_quantile_agreement',
            relative_errors=errors,relative_tolerance=.01,
            total_method_seconds=prep['preparation_seconds']+report['prepared_model_wall_seconds'])
    except BaseException as exc:
        report.update(status='failed_execution',error=repr(exc));raise
    finally:
        monitor.close();report.update(resources=monitor.snapshot(),process_seconds=time.perf_counter()-START)
        save_json(output,report)
    print(report['status'],report.get('limits'),report.get('relative_errors'),flush=True)


if __name__=='__main__':main()
