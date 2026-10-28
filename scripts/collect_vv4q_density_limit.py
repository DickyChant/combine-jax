"""Summarize completed corrected VV4q limits; no fitting or resubmission."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=Path('reports/asymptotic_vv4q_noVH_MX4000_density_limit.json')
    output=Path('reports/asymptotic_vv4q_noVH_MX4000_density_limit_comparison.json')
    assert not output.exists()
    run=json.loads(path.read_text())
    assert run['status']=='passed_quantiles' and run['float64']
    assert run['template_convention']=='v10_density'
    assert run['dataset']['mode']=='prefit' and run['dataset']['signal_strength']==0.
    assert all(p['status']==0 for p in run['profile_fits'])
    gate_path=Path('reports/asymptotic_vv4q_noVH_MX4000_density_gate.json')
    assert sha(gate_path)==run['gate_sha256']
    gate=json.loads(gate_path.read_text());assert gate['status']=='passed'
    rows=[dict(quantile=q,native_v10=b,combine_jax=a,absolute_error=abs(a-b),relative_error=abs(a-b)/abs(b))
          for q,a,b in zip(run['quantiles'],run['limits'],run['reference_limits'],strict=True)]
    assert max(r['relative_error'] for r in rows)<.01
    memory=max(m['total_bytes'] for m in run['compiler_memory'].values())
    assert memory<8*2**30
    result=dict(status='passed',script_sha256=sha(__file__),run_sha256=sha(path),gate_sha256=sha(gate_path),
        rows=rows,maximum_relative_error=max(r['relative_error'] for r in rows),
        workspace_asimov_preparation_seconds=run['preparation']['preparation_seconds'],
        tensor_setup_seconds=run['timings']['prepare_seconds'],lower_seconds=run['timings']['lower_seconds'],
        compile_seconds=run['timings']['compile_seconds'],execution_seconds=run['timings']['execution_seconds'],
        total_method_seconds=run['total_method_seconds'],native_seconds=run['native_wall_seconds'],
        observed_cross_host_wall_ratio=run['native_wall_seconds']/run['total_method_seconds'],
        same_node_speedup_certified=False,
        timing_scope='Local i7-7700HQ, affinity 0-7, with existing pip-v11 native tie-breaker on CPU0; dhi reference on b9p02p5453, one lxbatch core. This is a measured cross-host wall-time comparison, not a same-node speedup.',
        static_gib=memory/2**30,peak_rss_gib=run['resources']['peak_rss_mib']/1024,
        fits=len(run['profile_fits']),all_fits_converged=True,evaluation_counts=run['evaluation_counts'],
        validation=dict(max_nll_absolute_error=gate['max_nll_absolute_error'],
            max_gradient_relative_error=gate['max_gradient_relative_error'],
            generated_weights_identical_to_native_v10=gate['generated_weights_identical_to_native_v10']),
        environment=run['environment'],cpu_affinity=run['cpu_affinity'],cpu_model=run['cpu_model'],
        interpreter=sys.executable,versions={p:version(p) for p in ('jax','jaxlib','numpy','scipy','higgs-combine-tool','root')},
        version_scope='Collected from the unchanged project venv immediately after the run; no package installations occurred during the run.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
