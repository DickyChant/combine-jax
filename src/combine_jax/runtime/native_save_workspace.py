"""Native --saveWorkspace: the generated Asimov workspace plus a MultiDimFit snapshot.

Runs in the native stack (CMSSW Python 3.9 compatible), after the fit. Like
Combine's MultiDimFit --saveWorkspace, the snapshot holds every workspace
variable and category: fitted values for the floating parameters, unchanged
values (nominal global observables, frozen parameters) for the rest. Combine
can then start from the combine-jax best fit:

    combine -M MultiDimFit -d <output> -w w --snapshotName MultiDimFit -D <dataset> ...
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys


SNAPSHOT = 'MultiDimFit'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def save_workspace(R, generation, fit, output):
    """Add workspace `w` with the fitted snapshot to an existing output ROOT file."""
    if digest(generation['workspace']) != generation['workspace_sha256']:
        raise ValueError('Generated Asimov workspace changed after generation')
    values = fit['global_values']
    source = R.TFile.Open(generation['workspace'], 'READ')
    if not source or source.IsZombie():
        raise OSError(generation['workspace'])
    try:
        ws = source.Get('w')
        if not ws or not ws.data(generation['dataset']['name']):
            raise ValueError('Generated workspace lacks its Asimov dataset')
        if ws.getSnapshot(SNAPSHOT):
            raise ValueError('Generated workspace already has a '+SNAPSHOT+' snapshot')
        unchanged = {v.GetName(): float(v.getVal()) for v in ws.allVars() if v.GetName() not in values}
        for name, value in values.items():
            variable = ws.var(name)
            if not variable:
                raise ValueError('Fitted parameter is not a workspace variable: '+name)
            if not math.isfinite(value) or not variable.getMin() <= value <= variable.getMax():
                raise ValueError('Fitted value outside the workspace range: '+name)
            variable.setVal(value)
            if variable.getVal() != value:
                raise ValueError('Workspace variable did not take the fitted value: '+name)
        state = R.RooArgSet(ws.allVars())
        state.add(ws.allCats())
        ws.saveSnapshot(SNAPSHOT, state)
        target = R.TFile.Open(str(output), 'UPDATE')
        if not target or target.IsZombie():
            raise OSError(output)
        try:
            if target.Get('w'):
                raise ValueError('Output already holds a workspace')
            if target.WriteTObject(ws, 'w') <= 0:
                raise RuntimeError('Workspace write failed')
        finally:
            target.Close()
    finally:
        source.Close()
    check = R.TFile.Open(str(output), 'READ')
    try:
        saved = check.Get('w')
        if not saved or not check.Get('limit') or not saved.loadSnapshot(SNAPSHOT):
            raise RuntimeError('Saved workspace, snapshot or limit tree missing on reopen')
        for name, value in values.items():
            if saved.var(name).getVal() != value:
                raise RuntimeError('Snapshot value differs on reopen: '+name)
        for name, value in unchanged.items():
            if saved.var(name).getVal() != value:
                raise RuntimeError('Unfitted value differs on reopen: '+name)
        categories = {c.GetName(): int(c.getIndex()) for c in saved.allCats()}
    finally:
        check.Close()
    return dict(status='saved', snapshot=SNAPSHOT, workspace='w', dataset=generation['dataset']['name'],
        fitted_parameters=len(values), snapshot_variables=len(values)+len(unchanged),
        categories=categories, reopen_verified=True)


def main(generation_path, fit_path, output, report_path):
    import ROOT as R
    R.gROOT.SetBatch(True)
    base = Path(os.environ['CMSSW_BASE'])
    library = base/'lib'/os.environ['SCRAM_ARCH']/'libHiggsAnalysisCombinedLimit.so'
    generation = json.loads(Path(generation_path).read_text())
    if digest(library) != generation['library_sha256']:
        raise RuntimeError('Native library differs from the one that generated the Asimov')
    if R.gSystem.Load(str(library)) < 0:
        raise RuntimeError('Cannot load the configured native Combine library')
    report = save_workspace(R, generation, json.loads(Path(fit_path).read_text()), output)
    report.update(root_version=R.gROOT.GetVersion(), combine_version=generation['combine_version'])
    Path(report_path).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
