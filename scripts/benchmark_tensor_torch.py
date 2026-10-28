"""Synchronized PyTorch eager/CUDA-graph measurements on the neutral artifact."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu','gpu'), required=True)
    parser.add_argument('--mode', choices=('eager','cuda_graph'), default='eager')
    parser.add_argument('--operation', choices=('nll','value_gradient'), required=True)
    parser.add_argument('--repeat', type=int, default=26)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or (args.mode == 'cuda_graph' and args.backend != 'gpu'):
        parser.error('Positive repeat count and a GPU for CUDA graphs are required')
    numeric = args.output.with_suffix('.npz')
    assert not args.output.exists() and not numeric.exists()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    started = time.perf_counter()
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda:0' if args.backend == 'gpu' else 'cpu')
    if args.backend == 'gpu':
        assert torch.cuda.is_available()
        probe = torch.ones((), dtype=torch.float64, device=device)
        assert (probe+probe).item() == 2.
        kind = torch.cuda.get_device_name(device)
    else:
        kind = 'cpu'
    print('verified torch device', device, kind, 'float64', flush=True)
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.torch_backend import make_torch_function, to_torch_arrays
    from combine_jax.tensor.parameter_info import model_parameter_info
    report = dict(provenance(), status='running', stage='load', script_sha256=sha(__file__),
        python=sys.version, torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        backend=args.backend, mode=args.mode, operation=args.operation, float64=True,
        devices=[dict(platform=args.backend, kind=kind, id=0)],
        torch_intraop_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        model=str(args.model), model_sha256={n:sha(args.model/n) for n in ('model.json','arrays.npz')},
        points_source=str(args.points), points_source_sha256=sha(args.points), timings={},
        validation='Pending independent numerical audit',
        timing_scope='Synchronized device execution; points and model arrays resident, host output copies excluded. Warm calls cycle points 1..25,0. CUDA-graph calls include the required resident input copy before replay.',
        execution='PyTorch eager operators; no torch.compile. CUDA-graph mode captures the same operators and backward pass after three side-stream warmups.',
        documentation='https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs')
    monitor = Resources(args.backend == 'gpu', args.output)
    def synchronize():
        if args.backend == 'gpu':
            torch.cuda.synchronize(device)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started, resources=monitor.snapshot())
        if args.backend == 'gpu':
            report['torch_allocator'] = dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                scope='PyTorch allocator only; excludes driver/context and non-PyTorch allocations.')
        save_json(args.output, report)
    def stage(name):
        report['stage'] = monitor.stage = name
        save()
    save()
    try:
        if args.backend == 'gpu':
            free, total = torch.cuda.mem_get_info(device)
            fraction = min(.8, .8*free/total)
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            report['allocator_limit'] = dict(fraction_of_total=fraction, bytes=fraction*total,
                                             initial_free_bytes=free, total_bytes=total)
        before = time.perf_counter()
        model = TensorBundle.load(args.model)
        from tensor_benchmark_dataset import require_blind_model
        require_blind_model(model)
        info = model_parameter_info(model)
        with np.load(args.points, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['parameter_names'], model.metadata['parameter_names'])
            points = archive['points']
        assert points.dtype == np.float64 and points.ndim == 2 and len(points)
        assert points.shape[1] == len(model.metadata['parameter_names']) and np.isfinite(points).all()
        assert np.all(points >= info.arrays['lower']) and np.all(points <= info.arrays['upper'])
        report.update(parameters=points.shape[1], points=len(points), structure_key=model.structure_key)
        report['timings']['load_seconds'] = time.perf_counter()-before
        before = time.perf_counter()
        arrays = to_torch_arrays(model, device=device)
        requires_grad = args.operation == 'value_gradient'
        residents = [torch.tensor(point, dtype=torch.float64, device=device, requires_grad=requires_grad) for point in points]
        function = make_torch_function(model)
        synchronize()
        report['timings']['prepare_and_transfer_seconds'] = time.perf_counter()-before
        def invoke(point):
            value = function(point, arrays)
            return (value, torch.autograd.grad(value, point)[0]) if requires_grad else (value,)
        if args.mode == 'cuda_graph':
            stage('graph_warmup')
            static_input = residents[0].detach().clone().requires_grad_(requires_grad)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            before = time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    warm_result = invoke(static_input)
            torch.cuda.current_stream().wait_stream(stream)
            synchronize()
            del warm_result
            report['timings']['side_stream_warmup_seconds'] = time.perf_counter()-before
            stage('graph_capture')
            graph = torch.cuda.CUDAGraph()
            before = time.perf_counter()
            with torch.cuda.graph(graph):
                static_output = invoke(static_input)
            synchronize()
            report['timings']['graph_capture_seconds'] = time.perf_counter()-before
            def execute(point):
                with torch.no_grad():
                    static_input.copy_(point)
                graph.replay()
                return static_output
        else:
            execute = invoke
        stage('first_execute')
        before = time.perf_counter()
        output = execute(residents[0]); synchronize()
        report['timings']['first_execute_seconds'] = time.perf_counter()-before
        stage('warm')
        warm = []
        for i in range(args.repeat):
            before = time.perf_counter()
            output = execute(residents[(i+1) % len(residents)]); synchronize()
            warm.append(time.perf_counter()-before)
            report['timings'].update(warm_seconds=warm, warm_median_seconds=float(np.median(warm)))
            save()
        stage('capture')
        outputs = []
        for point in residents:
            current = execute(point); synchronize()
            outputs.append([value.detach().cpu().numpy().copy() for value in current])
        capture = dict(parameter_names=np.array(model.metadata['parameter_names']), points=points,
                       values=np.array([row[0] for row in outputs]))
        if requires_grad:
            capture['gradients'] = np.array([row[1] for row in outputs])
        assert all(np.isfinite(value).all() for key, value in capture.items() if key != 'parameter_names')
        np.savez_compressed(numeric, **capture)
        assert not any(name in sys.modules for name in ('ROOT','jax'))
        assert all(sha(p) == digest for p, digest in report['sources'].items())
        report.update(status='measured_pending_validation', stage='complete', arrays=str(numeric),
                      arrays_sha256=sha(numeric), root_imported=False, jax_imported=False)
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        monitor.close(); save()
    print(report['status'], report['timings'], flush=True)


if __name__ == '__main__':
    main()
