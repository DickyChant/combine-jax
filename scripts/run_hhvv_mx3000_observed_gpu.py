"""GPU phases of the explicitly unblinded combined MX3000 limit, and optional retry."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from hhvv_mx3000_authorization import authorize
from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['verify', 'background', 'observed', 'expected025'])
    args = p.parse_args()
    started = time.perf_counter()
    manifest_path = Path('build/hhvv_observed_runtime.json')
    manifest = json.loads(manifest_path.read_text())
    authorize(manifest)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu' and jax.config.x64_enabled
    assert any(x in devices[0].device_kind for x in ('A100','H100'))
    assert 'ROOT' not in sys.modules
    jax.config.update('jax_compilation_cache_dir', str(Path('build/gpu_compilation_cache').resolve()))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    print('GPU verified', devices[0].device_kind, jax.__version__, args.phase, flush=True)
    if args.phase == 'verify':
        for name, checksum in manifest['files'].items():
            assert sha(name) == checksum, name
        return
    suffix = {'background':'background_fit', 'observed':'observed_limit', 'expected025':'expected025_retry'}[args.phase]
    output = Path('reports')/('gpu_hhvv_MX3000_'+suffix+'.json')
    assert not output.exists()
    report = dict(provenance(), status='running', phase=args.phase, mass=3000,
        manifest_sha256=sha(manifest_path), source_revision=manifest['source_revision'],
        gpu=devices[0].device_kind, versions={n:version(n) for n in ('jax','jaxlib','jax-cuda12-plugin')},
        float64=True, static_guard_gib=8., template_convention='v10_density',
        fixed_discrete_choices=manifest['fixed_discrete_choices'], fit_traces={}, jax_allocator_samples=[])
    save_json(output, report)
    monitor = Resources(True, output)
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.limit_profile import LimitProfiler
    from combine_jax.methods.asymptotic import observed_limit, expected_nll_crossing
    from combine_jax.methods.contracts import validate_expected_model
    original = LimitProfiler._accept
    def progress(self, result, kind, mu):
        try:
            return original(self, result, kind, mu)
        finally:
            label = self.report_label
            report['fit_traces'][label] = list(self.records)
            report.setdefault('timings', {})[label] = dict(self.timings)
            report.setdefault('compiler_memory', {})[label] = self.compiler_memory
            report['jax_allocator_samples'].append(dict(stage=label, kind=kind, mu=mu, stats=devices[0].memory_stats()))
            report['resources'] = monitor.snapshot()
            save_json(output, report)
            print(label, kind, mu, self.records[-1], flush=True)
    LimitProfiler._accept = progress
    def load(kind):
        prep = json.loads(Path('reports/gpu_hhvv_MX3000_'+kind+'_import.json').read_text())
        assert prep['status'] == 'imported'
        for name, digest in prep['model_sha256'].items():
            assert sha(Path(prep['model'])/name) == digest
        model = TensorBundle.load(prep['model'])
        proof = model.metadata['mx3000_observed_authorization']
        assert proof['manifest_sha256'] == sha(manifest_path) and proof['mass'] == 3000
        assert model.metadata['provenance']['discrete_choices'] == manifest['fixed_discrete_choices']
        return model
    def profiler(model, label, expected=False):
        def compile_progress(name, stage, timings, memory):
            monitor.stage = label+':'+name+':'+stage
            event = dict(label=label, executable=name, stage=stage,
                         resources=monitor.snapshot(), wall_seconds=time.perf_counter()-started,
                         current_rss_mib=int(Path('/proc/self/statm').read_text().split()[1])
                             *os.sysconf('SC_PAGE_SIZE')/2**20)
            report.setdefault('compilation_events', []).append(event)
            report.setdefault('timings', {})[label] = timings
            report.setdefault('compiler_memory', {})[label] = memory
            save_json(output, report)
            print('compile', label, name, stage, event['resources'], flush=True)
        result = LimitProfiler(model, 'r_hhvv', compiler_memory_limit_gib=8.,
            global_up=None if expected else .5, global_start_fraction=.01 if expected else .1,
            compile_kinds=('conditional',) if label == 'background' else ('global', 'conditional'),
            release_compile_cache=True, compile_progress=compile_progress)
        result.report_label = label
        report.setdefault('timings', {})[label] = dict(result.timings)
        report.setdefault('compiler_memory', {})[label] = result.compiler_memory
        save_json(output, report)
        monitor.stage = label+':fitting'
        return result
    try:
        if args.phase == 'background':
            model = load('observed')
            fit = profiler(model, 'background')
            del model
            begin = time.perf_counter()
            fit(0.)
            assert fit.records[-1]['corrected_estimated_edm'] < 1e-4
            report.update(status='converged', fit=fit.records[-1], names=fit.names,
                point=fit.point().tolist(), poi_value=float(fit.point()[fit.index]),
                execution_seconds=time.perf_counter()-begin)
        elif args.phase == 'observed':
            model = load('observed')
            data = profiler(model, 'observed')
            del model
            model = load('conditional_asimov')
            validate_expected_model(model, 'r_hhvv', run='expected', discrete_policy='fixed')
            expected = profiler(model, 'conditional_asimov')
            # Start its global fit at the exact generating point, including the
            # conditional globals. Observed global start retains native 0.1*rMax.
            asimov_initial = expected.initial
            del model
            resident_d = sum(a.nbytes for a in data.arrays.values())
            resident_a = sum(a.nbytes for a in expected.arrays.values())
            total_static = max(max(m['total_bytes'] for m in data.compiler_memory.values())+resident_a,
                               max(m['total_bytes'] for m in expected.compiler_memory.values())+resident_d)
            report['combined_static_live_bytes'] = total_static
            assert total_static <= 8*2**30, 'Two live models exceed the unchanged 8 GiB static guard'
            begin = time.perf_counter()
            dmin = data.global_minimum()
            dhat = float(data.point()[data.index])
            data_global_point = data.warm
            report['global_fit'] = dict(data.records[-1], r_hhvv=dhat)
            save_json(output, report)
            amin = expected.global_minimum(initial=asimov_initial)
            asimov_global_point = expected.warm
            report['asimov_global_fit'] = expected.records[-1]
            report['search_attempts'] = []
            # A conditional fit finding a materially lower NLL invalidates the
            # denominator. Refit globally from that point, do not hide it in q=0.
            class LowerMinimum(RuntimeError):
                pass
            def checked(prof, baseline, mu):
                value = prof(mu)
                if value < baseline-.005:
                    raise LowerMinimum(prof.report_label)
                return value
            for attempt in range(3):
                try:
                    result = observed_limit(lambda mu: checked(data,dmin,mu),
                        lambda mu: checked(expected,amin,mu), dmin, amin, dhat, data.r_max)
                    report.update(result)
                    break
                except LowerMinimum as exc:
                    report['search_attempts'].append(dict(attempt=attempt, lower_minimum_dataset=str(exc)))
                    if attempt == 2:
                        raise
                    if str(exc) == 'observed':
                        dmin = data.global_minimum(initial=data.warm)
                        dhat = float(data.point()[data.index]); data_global_point = data.warm
                        report['global_fit'] = dict(data.records[-1], r_hhvv=dhat)
                    else:
                        amin = expected.global_minimum(initial=expected.warm)
                        asimov_global_point = expected.warm
                        report['asimov_global_fit'] = expected.records[-1]
                    data.warm, expected.warm = data_global_point, asimov_global_point
            records = data.records+expected.records
            assert all(r['status']==0 and r['corrected_estimated_edm'] < 1e-4 for r in records)
            report.update(status='computed_pending_native_validation', all_fits_converged=True,
                execution_seconds=time.perf_counter()-begin,
                iterations=sum(r['iteration'] for r in records),
                function_evaluations=sum(r['function_evaluations'] for r in records),
                gradient_evaluations=sum(r['gradient_evaluations'] for r in records),
                no_fit_asimov=False, native_limit_validation='pending',
                scope='Combined HHVV MX3000 only, explicitly unblinded; fixed nominal discrete choices.')
        else:
            # Run only after the observed report is durably saved. No median or
            # other expected bands are repeated. Outer controller caps 10 minutes.
            observed = json.loads(Path('reports/gpu_hhvv_MX3000_observed_limit.json').read_text())
            assert observed['status'] == 'computed_pending_native_validation'
            model = TensorBundle.load('build/hhvv_observed/prefit_model')
            validate_expected_model(model, 'r_hhvv', run='blind', discrete_policy='fixed')
            fit = profiler(model, 'prefit_expected025', expected=True)
            del model
            begin = time.perf_counter()
            baseline = fit.global_minimum()
            global_point = fit.warm
            target = expected_nll_crossing(.025)
            fit.warm = global_point
            at_zero = fit(0.)-baseline
            trace = [dict(mu=0., delta_nll=at_zero)]
            report.update(target_delta_nll=target, minimum_nll=baseline, scan=trace)
            if at_zero >= target:
                report.update(status='unresolved_baseline_offset', reason='Conditional r=0 already exceeds the target; a lower r bracket cannot repair this denominator.')
            else:
                lo, hi = 0., .13017654418945312
                value = fit(hi)-baseline; trace.append(dict(mu=hi, delta_nll=value))
                if value < target:
                    report.update(status='unresolved_upper_bracket')
                else:
                    for _ in range(20):
                        if time.perf_counter()-started > 540:
                            report['status'] = 'optional_time_budget'; break
                        mu = .5*(lo+hi)
                        value = fit(mu)-baseline; trace.append(dict(mu=mu, delta_nll=value))
                        if value < target: lo = mu
                        else: hi = mu
                        if hi-lo < max(1e-5,.005*mu):
                            report.update(status='bracketed', expected_quantile=.025,
                                limit=.5*(lo+hi), bracket=[lo,hi]); break
            report['execution_seconds'] = time.perf_counter()-begin
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        LimitProfiler._accept = original
        monitor.close()
        report.update(process_seconds=time.perf_counter()-started, resources=monitor.snapshot(),
            jax_allocator_final=devices[0].memory_stats(), root_imported='ROOT' in sys.modules)
        save_json(output, report)
    print(report['status'], report.get('observed_limit'), flush=True)


if __name__ == '__main__':
    main()
