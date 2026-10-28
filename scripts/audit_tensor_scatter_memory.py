"""Audit compiler allocations and measured calls for the unique-scatter change."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=Path('reports/tensor_hhvv_unique_scatter_audit.json');assert not output.exists()
    revision='212b7d1e328e3f9b78744fb093296805290b836b';profiles=[];artifacts={}
    for label in ('baseline','unique'):
        path=Path(f'reports/tensor_hhvv_memory_{label}.json');p=json.loads(path.read_bytes())
        assert p['status']=='complete' and p['float64'] and p['backend']=='cpu'
        assert p['script_sha256']==sha('scripts/profile_tensor_buffers.py')
        assert sha(p['import_report'])==p['import_report_sha256']
        raw=Path(f'reports/tensor_hhvv_memory_{label}_xla.txt')
        original=Path(f'build/tensor_hhvv_memory_{label}/xla/module_0000.jit_nll.cpu_after_optimizations-memory-usage-report.txt')
        assert sha(raw)==sha(original)
        artifacts[str(path)]=sha(path);artifacts[str(raw)]=sha(raw);profiles.append(p)
    old,new=profiles
    for key in ('import_report_sha256','jax_version','hostname','backend','float64','cpu_affinity'):assert old[key]==new[key]
    changed={p for p in old['tensor_sources'] if old['tensor_sources'][p]!=new['tensor_sources'][p]}
    assert changed=={'src/combine_jax/tensor/binned_channels.py'}
    for p,digest in old['tensor_sources'].items():
        blob=subprocess.run(['git','show',f'{revision}:{p}'],check=True,capture_output=True).stdout
        assert hashlib.sha256(blob).hexdigest()==digest
        if p in changed:
            previous=ast.parse(blob);current=ast.parse(Path(p).read_text())
            assert len(previous.body)==len(current.body)
            changed_definitions=[getattr(a,'name',None) for a,b in zip(previous.body,current.body,strict=True) if ast.dump(a)!=ast.dump(b)]
            assert changed_definitions==['make_binned_function']
    for p,digest in new['tensor_sources'].items():assert sha(p)==digest
    captures=[]
    for label,path in [('baseline',Path('reports/tensor_hhvv_gradients_capture.json')),
                       ('unique',Path('reports/tensor_hhvv_unique_gradients_capture.json'))]:
        r=json.loads(path.read_bytes());assert r['status']=='captured' and r['captured_points']==26
        assert sha(r['arrays'])==r['arrays_sha256'];artifacts[str(path)]=sha(path);captures.append(r)
    before,after=captures
    assert before['tensor_sources']==old['tensor_sources'] and after['tensor_sources']==new['tensor_sources']
    for key in ('import_report_sha256','source','jax_version','python','float64','backend','hostname','cpu_affinity','cpu_model','environment'):
        assert before[key]==after[key]
    for key,value in after['compiler_memory'].items():assert value==new['compiler_memory'][key]
    with np.load(before['arrays'],allow_pickle=False) as a,np.load(after['arrays'],allow_pickle=False) as b:
        for key in ('parameter_names','point','validation_points','validation_labels'):np.testing.assert_array_equal(a[key],b[key])
        difference=np.abs(a['validation_gradients']-b['validation_gradients'])
        relative=difference/np.maximum(np.abs(a['validation_gradients']),1e-8)
        assert np.isfinite(relative).all() and np.max(relative)<1e-4
        value_difference=float(np.max(np.abs(a['validation_vg_values']-b['validation_vg_values'])))
        assert value_difference<1e-6
    gate=Path('reports/tensor_hhvv_unique_gradients.json');g=json.loads(gate.read_bytes());assert g['status']=='passed'
    assert g['capture_sha256']==artifacts['reports/tensor_hhvv_unique_gradients_capture.json'];artifacts[str(gate)]=sha(gate)
    values=Path('reports/tensor_hhvv_unique_values_audit.json');v=json.loads(values.read_bytes());assert v['status']=='passed'
    assert set(v['runtime_source_updates'])==changed;artifacts[str(values)]=sha(values)
    testlog=Path('reports/tensor_unique_scatter_tests.log');assert '125 passed, 2 warnings' in testlog.read_text();artifacts[str(testlog)]=sha(testlog)
    old_temp=old['compiler_memory']['temp_size_in_bytes'];new_temp=new['compiler_memory']['temp_size_in_bytes']
    counts={label:sum('u32[5156236]' in line for line in Path(f'reports/tensor_hhvv_memory_{label}_xla.txt').read_text().splitlines()) for label in ('baseline','unique')}
    report=dict(status='passed',script_sha256=sha(__file__),baseline_revision=revision,artifacts=artifacts,
        changed_tensor_sources=sorted(changed),changed_definitions=changed_definitions,same_model_arrays=True,compiler_temporary_bytes=dict(baseline=old_temp,unique=new_temp),
        compiler_temporary_reduction_fraction=1-new_temp/old_temp,full_work_array_integer_allocation_entries=counts,
        baseline_peak_rss_mib=before['peak_rss_mib'],unique_peak_rss_mib=after['peak_rss_mib'],
        baseline_timings=before['timings'],unique_timings=after['timings'],
        warm_ratio_unique_over_baseline=after['timings']['median_warm_seconds']/before['timings']['median_warm_seconds'],
        between_tensor_runs=dict(gradient_max_absolute_error=float(np.max(difference)),gradient_max_relative_error=float(np.max(relative)),nll_max_absolute_error=value_difference),
        scope='Same local CPU, JAX version, affinity, runtime flags, artifact and points; only binned scatter uniqueness hints change numerical code. Two measured runs, not a controlled load study or GPU/Combine speed comparison.',
        guarantee='The existing validator rejects repeated fast, density-program and NLL output destinations before constructing the function. Runtime arrays must retain the validated index contract.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('passed',old_temp,new_temp,report['warm_ratio_unique_over_baseline'],flush=True)


if __name__=='__main__':main()
