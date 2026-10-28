"""Native Combine v11 per-call timings at exactly the tensor benchmark points."""
import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('build/tensor_benchmark_inputs/manifest.json'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.repeat > 0
    numeric_path = args.output.with_suffix('.npz')
    assert not args.output.exists() and not numeric_path.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    # Native reference runs must use the v11 wheel stack, not the LCG view's ROOT.
    assert not any('/lcg/views/' in p for p in sys.path)
    import numpy as np
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.parameter_info import model_parameter_info
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    started = time.perf_counter()
    case = json.loads(args.manifest.read_text())[args.case]
    for p, digest in case['hashes'].items():
        assert sha(p) == digest, p
    if 'workspace_sha256' in case:
        assert sha(case['workspace']) == case['workspace_sha256']
    model = TensorBundle.load(case['model'])
    from tensor_benchmark_dataset import benchmark_dataset
    dataset_name = benchmark_dataset(case, model, label=args.case)
    info = model_parameter_info(model)
    with np.load(case['points'], allow_pickle=False) as a:
        points = a['points']
        np.testing.assert_array_equal(a['parameter_names'], info.metadata['names'])
    with np.load(case['reference'], allow_pickle=False) as a:
        np.testing.assert_array_equal(a['points'], points)
        expected = a['values']
    assert len(points) > 1 and all(not np.array_equal(p, q) for p, q in zip(points, np.roll(points, 1, axis=0)))
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
    root.RooAbsReal.clearEvalErrorLog()
    prefix = Path(sys.prefix).resolve()
    native_libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(s in line for s in ('/ROOT/', '/root.libs/', '/cppyy', 'libHiggsAnalysisCombinedLimit'))
        and '.so' in line})
    assert native_libraries and all(Path(p).is_relative_to(prefix) for p in native_libraries)
    assert any('libCore.so' in p for p in native_libraries)
    assert any('libHiggsAnalysisCombinedLimit' in p for p in native_libraries)
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    report = dict(provenance(), status='running', stage='construct', python=sys.version,
        script_sha256=sha(__file__), case=args.case, manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        workspace=case['workspace'], workspace_sha256=sha(case['workspace']),
        root_version=str(root.gROOT.GetVersion()), combine_version=version('higgs-combine-tool'),
        python_prefix=str(prefix), native_libraries=native_libraries, runtime_flags=REFERENCE_FLAGS,
        configuration=case['configuration'], dataset=dataset_name, timings={},
        scope='Native Combine CachingSimNLL per-call only. This does not measure a Combine executable fit.',
        timing_policy='Prebuilt C++ parameter vectors. C++ steady_clock separates parameter updates and getVal; Python wall covers their call together. Each warm call changes the full parameter point. No graph import or JAX in this process.')
    monitor = Resources(False, args.output)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started, resources=monitor.snapshot())
        save_json(args.output, report)
    save()
    try:
        assert root.gInterpreter.Declare(r'''
            #include "RooRealVar.h"
            #include "RooAbsReal.h"
            #include <chrono>
            #include <vector>
            namespace combine_jax_tensor_benchmark {
            struct Objective {
                std::vector<RooRealVar*> variables;
                RooAbsReal* nll;
                double updateSeconds=0, evaluateSeconds=0;
                Objective(std::vector<RooRealVar*> v, RooAbsReal* n): variables(v), nll(n) {}
                double evaluate(std::vector<double> const& point) {
                    if (point.size()!=variables.size()) throw std::invalid_argument("Wrong parameter count");
                    auto start=std::chrono::steady_clock::now();
                    for (size_t i=0; i<point.size(); ++i)
                        if (variables[i]->getVal()!=point[i]) variables[i]->setVal(point[i]);
                    auto updated=std::chrono::steady_clock::now();
                    double value=nll->getVal();
                    auto end=std::chrono::steady_clock::now();
                    updateSeconds=std::chrono::duration<double>(updated-start).count();
                    evaluateSeconds=std::chrono::duration<double>(end-updated).count();
                    return value;
                }
            };
            }
        ''')
        with open_workspace(case['workspace']) as workspace, ExitStack() as stack:
            for name, config in case['configuration'].items():
                workspace.var(name).setVal(config['value'])
                workspace.var(name).setConstant(config['constant'])
            for config in model.metadata['provenance']['discrete_choices'].values():
                workspace.cat(config['category']).setIndex(config['index'])
            if 'fit_policy' in info.metadata:
                from combine_jax.tensor.fit_policy import check_native_fit_state, combine_v11_fit_context
                report['fit_state']=check_native_fit_state(workspace,model)
                stack.enter_context(combine_v11_fit_context(workspace))
            variables = [workspace.var(n) for n in info.metadata['names']]
            for variable, value in zip(variables, points[0], strict=True):
                variable.setVal(float(value))
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data(dataset_name), config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint()
            nll.clearZeroPoint()
            objective = root.combine_jax_tensor_benchmark.Objective(variables, nll)
            cpp_points = [root.std.vector('double')(point) for point in points]
            report['stage'] = monitor.stage = 'first_execute'
            save()
            before = time.perf_counter()
            first = float(objective.evaluate(cpp_points[0]))
            report['timings']['first_execute_seconds'] = time.perf_counter()-before
            assert abs(first-expected[0]) < 1e-6
            report['stage'] = monitor.stage = 'warm'
            warm = []
            for i in range(args.repeat):
                index = (i+1) % len(points)
                before = time.perf_counter()
                value = float(objective.evaluate(cpp_points[index]))
                wall = time.perf_counter()-before
                warm.append(dict(point=index, wall_seconds=wall, update_seconds=float(objective.updateSeconds),
                    evaluate_seconds=float(objective.evaluateSeconds), value=value))
                assert abs(value-expected[index]) < 1e-6
            report['timings'].update(warm=warm, wall_median_seconds=float(np.median([w['wall_seconds'] for w in warm])),
                update_median_seconds=float(np.median([w['update_seconds'] for w in warm])),
                evaluate_median_seconds=float(np.median([w['evaluate_seconds'] for w in warm])))
            report['stage'] = monitor.stage = 'capture'
            save()
            values = np.array([objective.evaluate(p) for p in cpp_points])
            errors = np.abs(values-expected)
            report['comparison'] = dict(points=len(points), max_nll_absolute_error=float(np.max(errors)),
                tolerance=1e-6, roofit_evaluation_errors=int(root.RooAbsReal.numEvalErrors()),
                passed=bool(np.isfinite(values).all() and np.all(errors < 1e-6) and root.RooAbsReal.numEvalErrors() == 0))
            np.savez_compressed(numeric_path, parameter_names=np.array(info.metadata['names']), points=points, values=values)
            report.update(arrays=str(numeric_path), arrays_sha256=sha(numeric_path))
            assert report['comparison']['passed']
            nll.setAnalyticBarlowBeeston(False)
            del objective,nll
        report.update(status='measured_native_validated', stage='complete', jax_imported='jax' in sys.modules)
        assert not report['jax_imported']
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        monitor.close()
        save()
    print(report['status'], report.get('comparison'), flush=True)


if __name__ == '__main__':
    main()
