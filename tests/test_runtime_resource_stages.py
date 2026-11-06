"""Process attribution stays distinct from device memory, including unavailable NVML values."""
import ctypes as C
import os
import threading
import pytest

from combine_jax.runtime.resources import Resources


def test_stage_peaks_do_not_attribute_other_process_memory():
    class NVML:
        own=20
        def nvmlDeviceGetMemoryInfo(self,handle,out):
            m=C.cast(out,C.POINTER(Resources.Memory)).contents
            m.total=1000;m.free=700;m.used=300
            return 0
        def nvmlDeviceGetComputeRunningProcesses_v2(self,handle,count,processes):
            C.cast(count,C.POINTER(C.c_uint)).contents.value=2
            processes[0].pid=os.getpid();processes[0].usedGpuMemory=self.own
            processes[1].pid=os.getpid()+1;processes[1].usedGpuMemory=280
            return 0
    r=Resources.__new__(Resources);r.nvml=NVML();r.handle=None;r.lock=threading.Lock()
    r.samples=r.device_peak=r.process_peak=0;r.pid_seen=False;r.stage_peaks={};r.stage='compile'
    r.sample_gpu();r.stage='execute';r.nvml.own=2**64-1;r.sample_gpu()
    assert r.stage_peaks['compile']['process_peak_bytes']==20
    assert r.stage_peaks['execute']['process_peak_bytes'] is None
    assert r.stage_peaks['execute']['device_peak_bytes']==300
    assert r.process_peak==20 and r.device_peak==300


@pytest.mark.parametrize('process_supported',[True,False])
def test_mig_sampling_uses_assigned_slice_handle_not_parent(monkeypatch,tmp_path,process_supported):
    uuid='MIG-12345678-1234-1234-1234-123456789abc'
    class NVML:
        def nvmlInit_v2(self):return 0
        def nvmlDeviceGetHandleByUUID(self,name,out):
            assert name==uuid.encode()
            C.cast(out,C.POINTER(C.c_void_p)).contents.value=77
            return 0
        def nvmlDeviceGetUUID(self,handle,out,size):
            assert handle.value==77;out.value=uuid.encode();return 0
        def nvmlDeviceGetName(self,handle,out,size):
            assert handle.value==77;out.value=b'H100 MIG';return 0
        def nvmlSystemGetDriverVersion(self,out,size):out.value=b'test';return 0
        def nvmlDeviceGetMemoryInfo(self,handle,out):
            assert handle.value==77
            m=C.cast(out,C.POINTER(Resources.Memory)).contents
            m.total=12*2**30;m.used=500*2**20;m.free=m.total-m.used
            return 0
        def nvmlDeviceGetComputeRunningProcesses_v2(self,handle,count,processes):
            assert handle.value==77
            if not process_supported:return 3
            C.cast(count,C.POINTER(C.c_uint)).contents.value=0
            return 0
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',uuid)
    monkeypatch.setattr(C,'CDLL',lambda _:NVML())
    sampler=Resources(True,tmp_path/'result.json')
    try:
        report=sampler.snapshot()['gpu']
        assert report['device_uuid']==uuid and report['device_total_mib']==12*1024
        assert report['device_peak_mib']==500 and report['process_peak_mib'] is None
        assert report['device_metadata']['memory_scope']=='assigned MIG slice'
        assert report['process_query_status']==('supported' if process_supported else 'not_supported')
    finally:sampler.close()


def test_host_stage_current_residency_is_not_the_lifetime_peak(monkeypatch):
    import combine_jax.runtime.resources as module
    values=iter([dict(rss_mib=120.,rusage_peak_rss_mib=150.,threads=4),
                 dict(rss_mib=80.,rusage_peak_rss_mib=150.,threads=2)])
    monkeypatch.setattr(module,'host_memory',lambda:next(values))
    r=Resources.__new__(Resources);r.lock=threading.Lock();r.host_stage_peaks={}
    r.stage='compile';r.sample_host();r.stage='execute';r.sample_host()
    assert r.host_stage_peaks['execute']['rss_mib']==80.
    assert r.host_stage_peaks['execute']['rusage_peak_rss_mib']==150.
    assert r.host_stage_peaks['compile']['threads']==4


def test_host_guard_names_stage_and_preserves_evidence(tmp_path):
    import json
    import subprocess
    import sys
    output=tmp_path/'fit.json'
    script='''import sys,time
from combine_jax.runtime.resources import Resources
monitor=Resources(False,sys.argv[1],rss_limit_mib=1)
monitor.stage='compile_refit'
time.sleep(2)
raise RuntimeError('Guard failed to stop the isolated child')
'''
    result=subprocess.run([sys.executable,'-c',script,str(output)],capture_output=True,text=True,timeout=10)
    assert result.returncode==75
    assert 'Host RSS guard exceeded during compile_refit' in result.stdout
    assert '--host-memory-limit-gib' in result.stdout
    guard=json.loads(output.with_name('fit_guard.json').read_text())
    assert guard['stage']=='compile_refit' and guard['limit_mib']==1
    assert guard['peak_rss_mib']>guard['limit_mib']
    assert guard['host_stage_peaks']['compile_refit']['samples']>0
    assert guard['resources']['host_memory']['rusage_peak_rss_mib']>guard['limit_mib']


@pytest.mark.parametrize('limit',[0.,-1.,float('nan'),float('inf')])
def test_host_guard_cannot_be_disabled_by_invalid_limits(limit,tmp_path):
    with pytest.raises(ValueError,match='positive and finite'):
        Resources(False,tmp_path/'result.json',rss_limit_mib=limit)
