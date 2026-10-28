"""Compare unfitted VV4q workspace configurations; never read observed data."""
import json
from pathlib import Path
from collections import Counter

from tensor_benchmark_common import sha, save_json
from combine_jax.root_io import open_workspace
from combine_jax.graph import walk


def inspect(path):
    print('opening', str(path), flush=True)
    with open_workspace(path) as ws:
        print('workspace opened', flush=True)
        mc = ws.obj('ModelConfig')
        config = {}
        for name in ('MX', 'MH', 'MHSM', 'MY', 'r_hhvv'):
            v = ws.var(name)
            if v:
                config[name] = dict(value=float(v.getVal()), constant=bool(v.isConstant()),
                    minimum=float(v.getMin()) if v.hasMin() else None,
                    maximum=float(v.getMax()) if v.hasMax() else None)
        ws.var('MX').setVal(4000.)
        ws.var('MX').setConstant(True)
        if ws.var('MH'):
            ws.var('MH').setVal(125.)
            ws.var('MH').setConstant(True)
        ws.var('r_hhvv').setVal(1.)
        nodes = walk(mc.GetPdf())
        print('graph walked', len(nodes), flush=True)
        graph = {n: dict(class_name=str(v.ClassName()), servers=[str(c.GetName()) for c in v.servers()])
                 for n, v in nodes.items()}
        observables = {str(v.GetName()) for v in mc.GetObservables()}
        variables = {}
        for name, v in nodes.items():
            if v.ClassName() in ('RooRealVar', 'RooConstVar') and name not in observables:
                variables[name] = dict(value=float(v.getVal()), constant=bool(v.isConstant()))
                if v.ClassName() == 'RooRealVar':
                    variables[name].update(minimum=float(v.getMin()) if v.hasMin() else None,
                        maximum=float(v.getMax()) if v.hasMax() else None)
        scales = {str(f.GetName()): float(f.getVal()) for f in ws.allFunctions()
                  if str(f.GetName()).startswith(('HHVV_', 'BR', 'r_DY_', 'r_VBF_'))}
        print('scalars read', len(scales), flush=True)
        return dict(path=str(path), sha256=sha(path), configuration_before_overrides=config,
            classes=dict(Counter(str(v.ClassName()) for v in nodes.values())),
            graph=graph, variables=variables, theory_scales_at_MX4000_r1=scales)


def main():
    local = Path('build/asymptotic_vv4q_inputs/noVH_4000/workspace.root')
    native = Path('/eos/cms/store/group/phys_b2g/sqian/crab_inference_dev/store/CreateWorkspace/hhvv_model__model_default/datacards_1b175c5dad/m125.0/VV4q_VHtest_noVH/workspace.root')
    native_copy = Path('build/dhi_noVH_MX4000_job/workspace.root')
    assert sha(native_copy) == sha(native)
    a, b = inspect(local), inspect(native_copy)
    report = dict(status='compared', script_sha256=sha(__file__),
        scope='Unfitted CreateWorkspace outputs only. No dataset accessed. Both inspected in isolated Combine v11/ROOT 6.40 after MX4000, MH125 and r=1 overrides.',
        workspaces={k:{n:v for n,v in x.items() if n not in ('graph','variables')} for k,x in [('local',a),('dhi',b)]}, differences={})
    for field in ('graph', 'variables', 'theory_scales_at_MX4000_r1'):
        report['differences'][field] = {n:dict(local=a[field].get(n), dhi=b[field].get(n))
            for n in sorted(a[field].keys() | b[field].keys()) if a[field].get(n) != b[field].get(n)}
    save_json(Path('reports/asymptotic_vv4q_noVH_MX4000_workspace_audit.json'),report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
