"""Reproduce bound cycling on analytic quadratics without changing the backend."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from tensor_benchmark_common import save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-cases',type=int,default=100)
    parser.add_argument('--all-cases',action='store_true',
                        help='Evaluate every prototype on every predefined case; do not stop at a reproducer')
    parser.add_argument('--reduced-space-prototype',action='store_true')
    args=parser.parse_args();assert not args.output.exists() and args.maximum_cases>0
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    revision='c8b9072d183c8ce72232307e38865f453cd85bbe'
    source=subprocess.check_output(['git','show',revision+':src/combine_jax/tensor/optimize.py'],text=True)
    original="            p = jnp.where(((x <= lower) & (p < 0)) | ((x >= upper) & (p > 0)), 0., p)"
    assert source.count(original)==1
    tolerance="            eps = 32 * jnp.finfo(initial.dtype).eps * (1 + jnp.abs(x))\n"
    near=tolerance+"            p = jnp.where(((x <= lower+eps) & (p < 0)) | ((x >= upper-eps) & (p > 0)), 0., p)"
    active=tolerance+"            blocked = ((x <= lower+eps) & (g > 0)) | ((x >= upper-eps) & (g < 0))\n"+"            p = jnp.where(blocked | ((x <= lower+eps) & (p < 0)) | ((x >= upper-eps) & (p > 0)), 0., p)"
    replacements=dict(original=original,consistent_boundary_tolerance=near,freeze_blocked_rows=active)
    sources={name:source.replace(original,replacement) for name,replacement in replacements.items()}
    reduced_edits={}
    if args.reduced_space_prototype:
        reduced_edits={
            "            p = -s['inverse'] @ pg": tolerance+
                "            blocked = ((x <= lower+eps) & (g > 0)) | ((x >= upper-eps) & (g < 0))\n"+
                "            metric = jnp.where(blocked[:, None] | blocked[None, :], identity, s['inverse'])\n"+
                "            p = -metric @ pg",
            original: near,
            "            base = jnp.where(descent, s['inverse'], identity)":
                "            base = jnp.where(descent, metric, identity)",
            "            step, change = candidate-x, newgradient-g":
                "            step, change = candidate-x, jnp.where(blocked, 0., newgradient-g)",
        }
        reduced=source
        for before,after in reduced_edits.items():
            assert reduced.count(before)==1
            reduced=reduced.replace(before,after)
        sources['reduced_space_curvature']=reduced
    def objective(x,a):
        delta=x-a['minimum']
        return a['offset']+.5*delta@a['matrix']@delta+a['linear']@delta
    functions={}
    for name,prototype_source in sources.items():
        namespace={};exec(compile(prototype_source,'<frozen_optimizer_'+name+'>','exec'),namespace)
        functions[name]=jax.jit(namespace['make_bfgs'](objective,max_iterations=1000,gradient_tolerance=1e-7))
    rng=np.random.default_rng(236)
    initial=np.zeros(6);lower=np.array([-10.,-10.,-10.,-np.inf,-np.inf,-np.inf]);upper=np.full(6,np.inf)
    report=dict(status='searching',script_sha256=sha(__file__),jax_version=jax.__version__,float64=True,
        frozen_source_revision=revision,frozen_optimizer_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_line=original,prototype_replacements=replacements,seed=236,checked_cases=0,
        all_cases=args.all_cases,maximum_cases=args.maximum_cases,case_results=[],
        reduced_space_edits=reduced_edits,
        prototype_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()},
        scope='Diagnostic source-derived prototypes on positive-definite analytic quadratics only. No production source changes, HHVV solver acceptance, or performance claims.')
    save_json(args.output,report)
    for case in range(args.maximum_cases):
        factor=rng.normal(size=(6,6));matrix=factor.T@factor+.1*np.eye(6)
        minimum=np.r_[np.full(3,-10.),rng.uniform(-2.,2.,size=3)]
        linear=np.r_[rng.uniform(.05,2.,size=3),np.zeros(3)]
        arrays=dict(matrix=matrix,minimum=minimum,linear=linear,offset=np.array(1.e7))
        baseline=jax.device_get(functions['original'](initial,arrays,lower,upper))
        report['checked_cases']=case+1
        qualifies=int(baseline['status'])==1 and np.max(abs(baseline['projected_gradient']))>=1e-4
        if not args.all_cases and not qualifies:continue
        results={'original':baseline}
        for name in sources.keys()-{'original'}:
            results[name]=jax.device_get(functions[name](initial,arrays,lower,upper))
        report['case_results'].append(dict(case_index=case,results={name:dict(
            status=int(result['status']),iterations=int(result['iteration']),
            projected_gradient_max=float(np.max(abs(result['projected_gradient']))),
            coordinate_error_max=float(np.max(abs(result['x']-minimum))),
            roundoff_accepts=int(result['roundoff_accepts']),
            skipped_updates=int(result['skipped_updates'])) for name,result in results.items()}))
        if args.all_cases:
            save_json(args.output,report)
            continue
        if int(results['freeze_blocked_rows']['status'])!=0:continue
        np.testing.assert_allclose(results['freeze_blocked_rows']['x'],minimum,atol=1e-6,rtol=0.)
        report.update(status='reproduced_analytic_bound_cycle',case_index=case,initial=initial,
            lower=lower,upper=upper,arrays=arrays,minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
            results={name:{k:v for k,v in result.items() if k!='inverse'} for name,result in results.items()})
        save_json(args.output,report)
        print(report['status'],case,[(name,int(r['status']),int(r['iteration']),float(np.max(abs(r['projected_gradient'])))) for name,r in results.items()],flush=True)
        return
    if args.all_cases:
        report['summary']={name:dict(
            converged=sum(c['results'][name]['status']==0 for c in report['case_results']),
            converged_with_coordinate_error_below_1e6=sum(
                c['results'][name]['status']==0 and c['results'][name]['coordinate_error_max']<=1e-6
                for c in report['case_results']),
            status_counts={str(status):sum(c['results'][name]['status']==status for c in report['case_results'])
                           for status in range(5)}) for name in functions}
        report['status']='completed_predefined_analytic_cases'
        save_json(args.output,report)
        print(report['status'],report['summary'],flush=True)
        return
    report['status']='no_reproduction_found';save_json(args.output,report)
    raise RuntimeError('No qualifying analytic reproducer found in the predeclared cases')


if __name__=='__main__':
    main()
