"""Framework-free provenance and resource sampling for tensor benchmark workers."""
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import threading
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not (-float('inf') < value < float('inf')):
        return None
    return value


def host_memory():
    """Current host residency and the kernel's lifetime peak; no model data read."""
    names={'VmRSS':'rss_mib','VmHWM':'lifetime_peak_rss_mib','RssAnon':'anonymous_mib',
           'RssFile':'file_backed_mib','VmData':'virtual_data_mib','VmSwap':'swap_mib'}
    result={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,text=line.partition(':')
        if key in names:result[names[key]]=float(text.split()[0])/1024
        elif key=='Threads':result['threads']=int(text)
    result['rusage_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return result


def save_json(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_group_memory(group, *, details=False):
    """Current RSS by process; optional PSS/private pages explain shared forks.

    Only process names and memory counters are read, never arguments or the
    environment. PSS is diagnostic; the existing aggregate RSS guard is unchanged.
    """
    result = []
    for item in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = item.read_text()
            fields = raw.rsplit(')', 1)[1].split()
            if int(fields[2]) != group:
                continue
            row = dict(pid=int(item.parent.name), parent_pid=int(fields[1]),
                name=raw.split('(', 1)[1].rsplit(')', 1)[0],
                rss_mib=int(fields[21])*os.sysconf('SC_PAGE_SIZE')/2**20)
            if details:
                try:
                    counters = {}
                    for line in (item.parent/'smaps_rollup').read_text().splitlines():
                        key, _, value = line.partition(':')
                        if key in ('Rss', 'Pss', 'Private_Clean', 'Private_Dirty'):
                            counters[key] = float(value.split()[0])/1024
                    row.update(pss_mib=counters.get('Pss'),
                        private_mib=counters.get('Private_Clean', 0.)+counters.get('Private_Dirty', 0.))
                except (OSError, ValueError) as exc:
                    row['smaps_error'] = type(exc).__name__
            result.append(row)
        except (OSError, ValueError, IndexError):
            continue
    return result


def provenance():
    from .cpu_threads import thread_configuration
    return dict(started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
        cpu_threads=thread_configuration(),
        workdir=str(Path.cwd()), cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')),
        environment={k: os.environ.get(k) for k in ('COMBINE_JAX_LCG_VIEW', 'CUDA_VISIBLE_DEVICES',
            'JAX_PLATFORMS', 'JAX_ENABLE_X64', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
            'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        sources={p.name: sha(p) for p in sorted((Path(__file__).parents[1]/'tensor').glob('*.py'))},
        common_script_sha256=sha(__file__))


def assigned_gpu_uuid(visible, candidates):
    """CUDA/Condor accepts UUID prefixes; NVML requires the complete UUID."""
    if not visible.startswith('GPU-'):
        raise ValueError(f'Expected a full-device GPU UUID or prefix: {visible!r}')
    matches = [uuid for uuid in candidates if uuid.startswith(visible)]
    if len(matches) != 1:
        raise RuntimeError(f'GPU assignment {visible!r} matches {len(matches)} physical devices')
    return matches[0]


class Resources:
    """Sample process RSS and the assigned GPU or MIG slice, including pools.

    GPU assignment follows CUDA_VISIBLE_DEVICES (UUID or physical index). A MIG
    UUID resolves to that slice's NVML handle, never its parent's memory counters.
    NVML device-wide memory includes other users; it is never called process memory.
    """
    class Memory(C.Structure):
        _fields_ = [(n, C.c_ulonglong) for n in ('total', 'free', 'used')]

    class Process(C.Structure):
        _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
                    ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]

    def __init__(self, gpu, report_path, rss_limit_mib=11264):
        if not math.isfinite(rss_limit_mib) or rss_limit_mib <= 0:
            raise ValueError('Host RSS limit must be positive and finite, in MiB')
        self.gpu = gpu
        self.path = Path(report_path)
        self.limit = rss_limit_mib
        self.started = time.perf_counter()
        self.stage = 'setup'
        self.samples = self.process_peak = self.device_peak = 0
        self.stage_peaks = {}
        self.host_stage_peaks = {}
        self.host_monitor_start = host_memory()
        self.pid_seen = False
        self.process_query_status = 'not_sampled'
        self.pid = os.getpid()
        self.namespace_pids = next((list(map(int, line.split()[1:]))
            for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('NSpid:')), [])
        self.error = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.device_metadata = None
        if gpu:
            self.nvml = C.CDLL('libnvidia-ml.so.1')
            self.check(self.nvml.nvmlInit_v2())
            self.handle = C.c_void_p()
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
            if visible.startswith('GPU-'):
                # CERN Condor supplies e.g. GPU-47e13020, which CUDA accepts but
                # nvmlDeviceGetHandleByUUID rejects with NVML_ERROR_NOT_FOUND.
                count = C.c_uint()
                self.check(self.nvml.nvmlDeviceGetCount_v2(C.byref(count)))
                candidates = []
                for index in range(count.value):
                    handle = C.c_void_p()
                    self.check(self.nvml.nvmlDeviceGetHandleByIndex_v2(C.c_uint(index), C.byref(handle)))
                    uuid = C.create_string_buffer(96)
                    self.check(self.nvml.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
                    candidates.append(uuid.value.decode())
                full_uuid = assigned_gpu_uuid(visible, candidates)
                self.check(self.nvml.nvmlDeviceGetHandleByUUID(full_uuid.encode(), C.byref(self.handle)))
            elif visible.startswith('MIG-'):
                # Condor supplies the full MIG UUID. Fail if it cannot be resolved:
                # using the physical parent would invalidate the slice memory guard.
                self.check(self.nvml.nvmlDeviceGetHandleByUUID(visible.encode(), C.byref(self.handle)))
            elif visible.isdecimal():
                self.check(self.nvml.nvmlDeviceGetHandleByIndex_v2(C.c_uint(int(visible)), C.byref(self.handle)))
            else:
                raise RuntimeError(f'Unsupported GPU assignment for memory attribution: {visible!r}')
            uuid = C.create_string_buffer(96)
            self.check(self.nvml.nvmlDeviceGetUUID(self.handle, uuid, len(uuid)))
            self.uuid = uuid.value.decode()
            if visible.startswith('MIG-'):
                # nvidia-smi --query-gpu does not accept MIG UUIDs on all drivers.
                name=C.create_string_buffer(256);driver=C.create_string_buffer(96)
                self.check(self.nvml.nvmlDeviceGetName(self.handle,name,len(name)))
                self.check(self.nvml.nvmlSystemGetDriverVersion(driver,len(driver)))
                self.device_metadata=dict(name=name.value.decode(),driver_version=driver.value.decode(),
                    uuid=self.uuid,memory_scope='assigned MIG slice')
            else:
                self.device_metadata = subprocess.check_output(['nvidia-smi', '-i', self.uuid,
                    '--query-gpu=name,driver_version,memory.total,uuid', '--format=csv,noheader'], text=True).strip()
            self.sample_gpu()
            self.baseline = self.device_peak
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    def sample_gpu(self):
        m = self.Memory()
        self.check(self.nvml.nvmlDeviceGetMemoryInfo(self.handle, C.byref(m)))
        self.device_total = int(m.total)
        count = C.c_uint(1024)
        processes = (self.Process * count.value)()
        code=self.nvml.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, C.byref(count), processes)
        if code==3:  # NVML_ERROR_NOT_SUPPORTED on some virtualized/MIG devices.
            own=None
            self.process_query_status='not_supported'
        else:
            self.check(code)
            self.process_query_status='supported'
            own = next((int(p.usedGpuMemory) for p in processes[:count.value] if p.pid == os.getpid()), None)
        if own == 2**64 - 1:
            own = None  # NVML_VALUE_NOT_AVAILABLE must not become an enormous measurement.
        with self.lock:
            self.samples += 1
            self.device_peak = max(self.device_peak, m.used)
            stage=self.stage_peaks.setdefault(self.stage,dict(device_peak_bytes=0,process_peak_bytes=None,samples=0))
            stage['samples']+=1
            stage['device_peak_bytes']=max(stage['device_peak_bytes'],int(m.used))
            if own is not None:
                self.pid_seen = True
                self.process_peak = max(self.process_peak, own)
                stage['process_peak_bytes']=max(stage['process_peak_bytes'] or 0,own)

    def sample_host(self):
        counters=host_memory()
        with self.lock:
            row=self.host_stage_peaks.setdefault(self.stage,dict(samples=0))
            row['samples']+=1
            for key,value in counters.items():row[key]=max(row.get(key,0),value)
        return counters

    def run(self):
        ticks = 0
        while not self.stop.wait(.05):
            ticks += 1
            counters=self.sample_host();rss=counters['rusage_peak_rss_mib']
            if rss > self.limit:
                with self.lock:
                    host_stages={k:dict(v) for k,v in self.host_stage_peaks.items()}
                guard = dict(status='terminated_resource_limit', stage=self.stage,
                    peak_rss_mib=rss, limit_mib=self.limit, elapsed_seconds=time.perf_counter()-self.started)
                guard.update(host_memory=counters,host_monitor_start=self.host_monitor_start,
                    host_stage_peaks=host_stages,
                    guard_basis='Lifetime process peak RSS, not current residency')
                try:
                    guard['resources']=self.snapshot()
                except Exception as exc:
                    # GPU accounting failure must not disable the host guard.
                    guard['resource_snapshot_error']=repr(exc)
                guard_path=self.path.with_name(self.path.stem + '_guard.json')
                save_json(guard_path, guard)
                print(f'Host RSS guard exceeded during {self.stage}: lifetime peak {rss:.1f} MiB, '
                    f'limit {self.limit} MiB, current {counters.get("rss_mib",float("nan")):.1f} MiB. '
                    f'Evidence saved in {guard_path}. Configure the public fit/scan limit with '
                    '--host-memory-limit-gib when the worker has sufficient allocated RAM.',flush=True)
                os._exit(75)
            if ticks % 400 == 0:
                print('heartbeat', self.stage, round(time.perf_counter()-self.started, 1), round(rss, 1), flush=True)
            if self.gpu:
                try:
                    self.sample_gpu()
                except Exception as exc:
                    self.error = repr(exc)
                    return

    def snapshot(self):
        counters=self.sample_host()
        result = dict(peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      rss_limit_mib=self.limit,host_memory=counters,
                      host_monitor_start=self.host_monitor_start,
                      guard_basis='Lifetime process peak RSS',host_sample_period_seconds=.05)
        with self.lock:result['host_stage_peaks']={k:dict(v) for k,v in self.host_stage_peaks.items()}
        if self.gpu:
            if not self.error:
                self.sample_gpu()
            with self.lock:
                result['gpu'] = dict(device_uuid=self.uuid, device_metadata=self.device_metadata,
                    process_id=self.pid, namespace_pids=self.namespace_pids, device_total_mib=self.device_total/2**20,
                    process_peak_mib=self.process_peak/2**20 if self.pid_seen else None,
                    process_seen=self.pid_seen, device_baseline_mib=self.baseline/2**20,
                    process_query_status=self.process_query_status,
                    device_peak_mib=self.device_peak/2**20, samples=self.samples,
                    stage_peaks={k:dict(v) for k,v in self.stage_peaks.items()},
                    period_seconds=.05, error=self.error,
                    scope='Sampled process resident memory includes retained pools and executables; device-wide values include other users.')
        return result

    def close(self):
        self.stop.set()
        self.thread.join()
