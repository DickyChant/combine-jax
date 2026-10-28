"""Audit a transferred GPU executable cache with independent native-value gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--native-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    submission=json.loads(args.submission.read_text())
    driver=json.loads(args.driver.read_text())
    native=json.loads(args.native_audit.read_text())
    previous_path=Path('reports/gpu_tensor_primary_phase2_cache_audit.json')
    previous=json.loads(previous_path.read_text())
    producer_submission_path=Path('reports/gpu_tensor_condor_phase2.json')
    producer=json.loads(producer_submission_path.read_text())
    assert producer['job_id']==submission['cache_producer_job'] and producer['job_id']!=submission['job_id']
    archive=Path(producer['cache_archive'])
    assert sha(archive)==producer['cache_archive_sha256']==submission['cache_input_sha256']
    assert previous['status']=='passed_fresh_process_gpu_reuse' and native['status']=='passed'
    files={}
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith('-cache'):
                relative=member.name.removeprefix('build/tensor_phase2_cache/')
                files[relative]=hashlib.file_digest(stream.extractfile(member),'sha256').hexdigest()
    assert len(files)==2 and files==driver['cache_before']
    records=[]
    for operation in ('nll','value_gradient'):
        old=next(row for row in previous['cases'] if row['operation']==operation)
        executable=[p for p in files if Path(p).name==old['executable_key']+'-cache']
        assert len(executable)==1 and files[executable[0]]==old['executable_sha256']
        old_path=Path(old['warm_report']);assert sha(old_path)==old['warm_report_sha256']
        old_report=json.loads(old_path.read_text())
        matched=[row for row in driver['commands'] if 'scripts/benchmark_tensor.py' in row['command']
                 and '--cache' in row['command'] and operation in row['command']]
        assert len(matched)==1
        command=matched[0];argv=command['command'];path=Path(argv[argv.index('--output')+1])
        assert command['status']=='passed' and command['returncode']==0
        assert native['benchmark_inputs'][str(path)]==sha(path)
        result=json.loads(path.read_text())
        assert result['status']=='measured_pending_validation' and not result['root_imported']
        source_evidence=verified_source_record(result,'scripts/benchmark_tensor.py',submission['patch_source_commit'])
        for key in ('operation','backend','mode','float64','model_sha256','structure_key',
                    'jax_version','jaxlib_version','points_source_sha256','parameters','points'):
            assert result[key]==old_report[key],key
        assert result['devices']==old_report['devices']
        assert result['environment']['COMBINE_JAX_LCG_VIEW']==old_report['environment']['COMBINE_JAX_LCG_VIEW']
        assert result['backend']=='gpu' and result['mode']=='jit' and result['points']==26
        assert sha(result['arrays'])==result['arrays_sha256']
        assert sha(old_report['arrays'])==old_report['arrays_sha256']
        log=Path(command['log']);text=log.read_text()
        hits=re.findall(r"Persistent compilation cache hit for 'jit_nll' with key '(jit_nll-[0-9a-f]+)'",text)
        snapshots=[row for row in driver['cache_snapshots'] if row['operation']==operation]
        assert len(snapshots)==1
        reused=hits==[old['executable_key']] and snapshots[0]['files']==files
        comparisons={}
        with np.load(old_report['arrays'],allow_pickle=False) as a,np.load(result['arrays'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                if key in ('parameter_names','points'):
                    np.testing.assert_array_equal(a[key],b[key]);continue
                delta=np.abs(a[key]-b[key])
                assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
                comparisons[key]=dict(bitwise_equal=bool(np.array_equal(a[key],b[key])),
                    maximum_absolute_difference=float(delta.max(initial=0.)),
                    maximum_relative_difference=float((delta/np.maximum(np.abs(a[key]),1e-8)).max(initial=0.)))
        records.append(dict(operation=operation,reused=bool(reused),cache_hit_keys=hits,
            expected_executable_key=old['executable_key'],expected_executable_sha256=old['executable_sha256'],
            cache_files_unchanged=snapshots[0]['files']==files,
            old_report=str(old_path),old_report_sha256=sha(old_path),report=str(path),report_sha256=sha(path),
            log=str(log),log_sha256=sha(log),source_evidence=source_evidence,
            old_host=old_report['host'],host=result['host'],device=result['devices'],
            old_driver=old_report['resources']['gpu']['device_metadata'],driver=result['resources']['gpu']['device_metadata'],
            old_uuid=old_report['resources']['gpu']['device_uuid'],uuid=result['resources']['gpu']['device_uuid'],
            lower_seconds=result['timings']['lower_seconds'],
            compile_or_cache_load_seconds=result['timings']['compile_or_cache_load_seconds'],
            warm_median_seconds=result['timings']['warm_median_seconds'],comparisons=comparisons))
    report=dict(status='passed_cross_job_gpu_cache_reuse' if all(r['reused'] for r in records) else 'cache_miss_numerically_valid',
        script_sha256=sha(__file__),producer_job=producer['job_id'],consumer_job=submission['job_id'],
        submission=str(args.submission),submission_sha256=sha(args.submission),
        producer_submission=str(producer_submission_path),producer_submission_sha256=sha(producer_submission_path),
        transferred_cache_sha256=sha(archive),driver=str(args.driver),driver_sha256=sha(args.driver),
        previous_audit=str(previous_path),previous_audit_sha256=sha(previous_path),
        native_audit=str(args.native_audit),native_audit_sha256=sha(args.native_audit),cases=records,
        scope='Separate Condor jobs, transferred executable checksums, explicit outer-NLL cache-hit logs and all-26-point native numerical gates. Same GPU architecture and JAX version; tracing/lowering still required. A cache miss is recorded explicitly, not counted as reuse.')
    save_json(args.output,report);print(report['status'],[(r['operation'],r['compile_or_cache_load_seconds']) for r in records],flush=True)


if __name__=='__main__':
    main()
