"""CPU scheduling configuration applied before importing numerical libraries."""
import os
import shlex
import sys

_configuration = None
ENVIRONMENT = ('COMBINE_JAX_CPU_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'XLA_FLAGS')


def positive_threads(value):
    if isinstance(value, bool) or str(value).strip() != str(int(value)) or int(value) < 1:
        raise ValueError('CPU threads must be a positive integer')
    return int(value)


def add_threads_option(parser):
    parser.add_argument('--threads', type=positive_threads,
        help='CPU only: select N allowed logical CPUs and configure thread pools before JAX starts; '
             'overrides COMBINE_JAX_CPU_THREADS')


def thread_configuration():
    """Report requested scheduling separately from actual process thread counts."""
    return dict(_configuration or dict(requested=None, source='environment_preserved'),
        affinity=sorted(os.sched_getaffinity(0)),
        environment={name: os.environ.get(name) for name in ENVIRONMENT})


def configure_cpu_threads(device, threads=None):
    global _configuration
    if device != 'cpu':
        if threads is not None:
            raise ValueError('--threads requires --device cpu / --backend cpu')
        return thread_configuration()
    source = 'cli' if threads is not None else 'COMBINE_JAX_CPU_THREADS'
    value = threads if threads is not None else os.environ.get('COMBINE_JAX_CPU_THREADS')
    if value is None:
        return thread_configuration()
    count = positive_threads(value)
    available = sorted(os.sched_getaffinity(0))
    if count > len(available):
        raise ValueError(f'Requested {count} CPU threads but affinity allows only {len(available)} logical CPUs; '
                         'allocate more CPUs in the scheduler or choose a smaller --threads value')
    if 'jax' in sys.modules or 'numpy' in sys.modules:
        raise RuntimeError('Set CPU threads in a fresh process before importing NumPy/JAX')
    # Affinity bounds the CPUs that XLA can use; a pool hint alone is insufficient.
    # Keep unrelated user XLA flags, overriding only these two CPU settings.
    flags = [flag for flag in shlex.split(os.environ.get('XLA_FLAGS', ''))
             if flag.split('=', 1)[0] not in
             ('--xla_cpu_multi_thread_eigen', 'intra_op_parallelism_threads', '--intra_op_parallelism_threads')]
    flags += [f'--xla_cpu_multi_thread_eigen={str(count > 1).lower()}',
              f'intra_op_parallelism_threads={count}']
    os.sched_setaffinity(0, available[:count])
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        os.environ[name] = str(count)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['COMBINE_JAX_CPU_THREADS'] = str(count)
    os.environ['XLA_FLAGS'] = shlex.join(flags)
    _configuration = dict(requested=count, source=source, affinity_before=available,
                          scope='Logical CPU affinity and pool configuration, not measured kernel concurrency')
    return thread_configuration()
