"""Require bit-identical native fit results with and without call instrumentation."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); assert not args.output.exists()
    rows=[]
    for case in ('counting','automcstats'):
        before=Path(f'reports/tensor_edm_refined_{case}_native.json')
        after=Path(f'reports/tensor_edm_counted_{case}_native.json')
        plain,counted=(json.loads(p.read_text()) for p in (before,after))
        assert plain['status']==counted['status']=='measured_fit_pending_agreement'
        assert plain['stopping_rule']==counted['stopping_rule']
        assert plain['input_workspace_sha256']==counted['input_workspace_sha256']
        for key in ('values','errors','edm','canonical_nll','status','edm_target_met'):
            np.testing.assert_array_equal(plain['fit'][key],counted['fit'][key])
        assert plain['evaluation_counts']['function']==counted['evaluation_counts']['function']
        instrumentation=counted['call_instrumentation']
        assert instrumentation['source_sha256']==sha('scripts/native_minuit_counters.cc')
        assert instrumentation['library_sha256']==sha(instrumentation['library'])
        for path,digest in instrumentation['capture']['files'].items():assert sha(path)==digest
        episodes=[e for e in instrumentation['capture']['events'] if e['event']=='minimize_return']
        assert len(episodes)==1
        e=episodes[0]
        # Nested single-argument -> two-argument -> DeltaGradient forwarding
        # must not multiply the scientific gradient count.
        assert e['phase_numerical_gradients']==4 and e['phase_hessian_gradients']==1
        assert e['gradient_entries']==[1,4,0,1,1],e
        assert e['phase_nll']==counted['evaluation_counts']['function']
        rows.append(dict(case=case,baseline=str(before),baseline_sha256=sha(before),
            instrumented=str(after),instrumented_sha256=sha(after),bit_identical_values_errors_edm_nll=True,
            native_function_calls=counted['evaluation_counts']['function'],
            native_numerical_gradient_invocations=counted['evaluation_counts']['numerical_gradient_evaluations'],
            nested_forwarding_deduplicated=True))
    save_json(args.output,dict(status='passed_instrumentation_preserves_native_fits',script_sha256=sha(__file__),cases=rows,
        scope='Same native stopping options, input workspace, fitted values, errors, EDM, canonical NLL and Nfcn. Counter scopes distinguish objective callbacks, actual NLL evaluations, ordinary numerical gradients and Hessian gradient checks. No performance claim.'))
    print('passed native instrumentation equivalence on counting and autoMCStats')


if __name__=='__main__':main()
