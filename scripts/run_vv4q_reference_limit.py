"""Blinded noVH MX4000 limit compared with the supplied independent dhi result."""
import time
START = time.perf_counter()
import json
import os
from pathlib import Path
import platform
import resource
import traceback

from tensor_benchmark_common import sha, save_json, Resources, provenance


def main():
    output = Path('reports/asymptotic_vv4q_noVH_MX4000.json')
    directory = Path('build/asymptotic_vv4q_noVH_MX4000')
    assert not output.exists() and not directory.exists()
    workspace = Path('build/asymptotic_vv4q_inputs/noVH_4000/workspace.root')
    card = workspace.with_name('card.txt')
    reference = Path('/eos/cms/store/group/phys_b2g/sqian/crab_inference_dev/store/UpperLimits/hhvv_model__model_default/datacards_1b175c5dad/m125.0/poi_r_hhvv/VV4q_VHtest_noVH/limit__poi_r_hhvv__params_MX4000.0.root')
    stage = json.loads(Path('reports/asymptotic_vv4q_workspace_preparation.json').read_text())
    record = next(x for x in stage['workspaces'] if x['variant']=='noVH' and x['mass']==4000)
    assert record['workspace_sha256']==sha(workspace) and record['source_card_sha256']==sha(card)
    report = dict(provenance(),status='preparing',script_sha256=sha(__file__),backend='cpu',
        workspace=str(workspace),workspace_sha256=sha(workspace),card_sha256=sha(card),
        reference=str(reference),reference_sha256=sha(reference),native_wall_seconds_user_reported=6.1*3600,
        comparison_scope='Independent dhi blinded expected limits; native user-reported 6.1 h on one lxbatch CPU core. Different host/core allocation; no same-node speed claim.',
        options=dict(run='expected',noFitAsimov=True,poi='r_hhvv',MH=125,MX=4000,MX_frozen=True),
        method_sources={str(p):sha(p) for p in Path('src/combine_jax/methods').glob('*.py')},
        profiles=[])
    save_json(output,report)
    monitor = Resources(False,output)
    try:
        import uproot
        # Read limit values only for expected quantiles. No observed entry is
        # retrieved, logged, or used if a file unexpectedly includes one.
        rows = []
        with uproot.open(reference) as handle:
            tree = handle['limit']
            for i,q in enumerate(tree['quantileExpected'].array(library='np')):
                if q>0:
                    value = tree['limit'].array(entry_start=i,entry_stop=i+1,library='np')[0]
                    rows.append((float(q),float(value)))
        report['reference_reader'] = dict(name='uproot',version=uproot.__version__,expected_entries_only=True)
        rows.sort()
        assert len(rows)==5 and all(abs(q-p)<1e-7 for (q,v),p in zip(rows,[.025,.16,.5,.84,.975],strict=True))
        report['reference_quantiles']=[q for q,v in rows]
        report['reference_limits']=[v for q,v in rows]
        save_json(output,report)
        from combine_jax.methods.workspace import prepare_expected_workspace
        from combine_jax.methods.runner import run_expected_model
        from combine_jax.tensor.limit_profile import LimitProfiler
        # A scalar progress callback changes no likelihood, fit or crossing
        # arithmetic; numerical sources and every completed fit are retained.
        accept = LimitProfiler._accept
        def progress(self,result,kind,mu):
            value = accept(self,result,kind,mu)
            report['profiles'].append(dict(self.records[-1]))
            save_json(output,report)
            print('profile',kind,mu,'iterations',self.records[-1]['iteration'],flush=True)
            return value
        LimitProfiler._accept = progress
        start = time.perf_counter()
        model,prep = prepare_expected_workspace(workspace,card,directory,poi='r_hhvv',
            run='expected',no_fit_asimov=True,parameters={'MX':4000},freeze=['MX'],mass=125)
        report.update(status='fitting',preparation=prep)
        save_json(output,report)
        result = run_expected_model(model,'r_hhvv')
        report.update(result,method_wall_seconds=time.perf_counter()-start)
        errors = [abs(a-b)/abs(b) for a,(_,b) in zip(result['limits'],rows,strict=True)]
        report.update(relative_errors=errors,relative_tolerance=.01,
            status='passed_quantiles' if max(errors)<.01 else 'failed_quantile_agreement')
    except BaseException as exc:
        report.update(status='failed_execution',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        monitor.close()
        report.update(resources=monitor.snapshot(),process_total_seconds=time.perf_counter()-START)
        save_json(output,report)
    print(report['status'],report['limits'],report['relative_errors'],flush=True)


if __name__=='__main__':
    main()
