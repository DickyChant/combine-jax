"""Compare a native CLI input with its saved workspace and clean snapshot."""
import argparse
import json
from pathlib import Path

from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    report=json.loads(args.native_fit.read_text());base=Path(report['workdir'])
    source=base/report['prepared_workspace']
    outputs=list((base/Path(report['prepared_workspace']).parent).glob('higgsCombine*.root'))
    assert len(outputs)==1 and sha(source)==report['prepared_workspace_sha256']
    from combine_jax.root_io import open_workspace
    def variables(collection):
        return {v.GetName():dict(value=float(v.getVal()),error=float(v.getError()),
            lower=float(v.getMin()),upper=float(v.getMax()),constant=bool(v.isConstant()),
            optimize_bounds=bool(v.getAttribute('optimizeBounds'))) for v in collection if v.InheritsFrom('RooRealVar')}
    records=[]
    for path in [source,outputs[0]]:
        with open_workspace(path) as workspace:
            snapshots={s.GetName():variables(s) for s in workspace.getSnapshots()}
            data=workspace.data('data_obs')
            records.append(dict(path=str(path),sha256=sha(path),variables=variables(workspace.allVars()),snapshots=snapshots,
                categories={v.GetName():v.getIndex() for v in workspace.allCats()},
                dataset=dict(name='data_obs',entries=int(data.numEntries()),sum_weights=float(data.sumEntries()))))
    before,after=records
    common=before['variables'].keys()&after['variables'].keys()
    changed_constants=[dict(name=n,before=before['variables'][n],after=after['variables'][n]) for n in sorted(common)
        if before['variables'][n]['constant'] and before['variables'][n]['value']!=after['variables'][n]['value']]
    clean=after['snapshots'].get('clean',{})
    starts=[dict(name=n,declared=v,actual_clean=clean[n]['value']) for n,v in zip(report['initial_parameters'],report['initial_values'],strict=True)
        if n in clean and clean[n]['value']!=v]
    result=dict(status='compared',script_sha256=sha(__file__),native_fit=str(args.native_fit),native_fit_sha256=sha(args.native_fit),
        workspaces=records,changed_constants=changed_constants,changed_imported_starts=starts,
        scope='Read-only saved workspace comparison; no new fit and no acceptance follows from matching a snapshot alone.')
    save_json(args.output,result)
    print('constants',[(r['name'],r['before']['value'],r['after']['value']) for r in changed_constants],flush=True)
    print('starts',starts,'snapshots',list(after['snapshots']),flush=True)


if __name__=='__main__':main()
