"""Native v10 observed limits on three synthetic fixtures only."""
import json
import os
from pathlib import Path
import subprocess
import time
import ROOT
from hhvv_mx3000_authorization import sha
assert ROOT.gROOT.GetVersion() == '6.30/07'
os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
base = Path('build/observed_generic_native_v10_rebound')
base.mkdir(exist_ok=False)
report = dict(combine_version='10.3.3', root_version=ROOT.gROOT.GetVersion(),
              scope='Synthetic cards only; no HHVV files.', comparisons=[])
for case in ('counting','automcstats','parametric'):
    directory = base/case
    directory.mkdir()
    workspace = Path('build/tensor_native_policy_inputs')/case/'workspace.root'
    source = ROOT.TFile.Open(str(workspace), 'READ')
    ws = source.Get('w')
    # ROOT 6.40 -> 6.30 ModelConfig streaming loses its workspace binding.
    # Rebind its named sets in a new copy, preserving PDFs and data.
    for mc_name in ('ModelConfig', 'ModelConfig_bonly'):
        mc = ws.obj(mc_name)
        if not mc: continue
        mc.SetWorkspace(ws)
        mc.SetParametersOfInterest(ws.set('POI'))
        mc.SetNuisanceParameters(ws.set('nuisances'))
        mc.SetObservables(ws.set('observables'))
        mc.SetGlobalObservables(ws.set('globalObservables'))
    workspace = directory/'workspace.root'
    ws.writeToFile(str(workspace))
    source.Close()
    command = ['combine','-M','AsymptoticLimits','-d',str(workspace.resolve()),'--run','observed','-m','125','-v','0']
    begin = time.monotonic()
    with (directory/'native.log').open('w') as stream:
        result = subprocess.run(command,cwd=directory,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    assert result.returncode == 0
    handle = ROOT.TFile.Open(str(next(directory.glob('higgsCombine*.root'))))
    rows = [(float(x.quantileExpected),float(x.limit)) for x in handle.Get('limit')]
    handle.Close()
    assert len(rows) == 1 and rows[0][0] == -1., rows
    report['comparisons'].append(dict(case=case,native_limit=rows[0][1],command=command,
        native_seconds=time.monotonic()-begin,workspace_sha256=sha(workspace),log_sha256=sha(directory/'native.log')))
    (base/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(case,rows,flush=True)
