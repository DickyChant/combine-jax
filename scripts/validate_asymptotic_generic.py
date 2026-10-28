"""Sequential small-card expected-limit comparisons against native Combine v11."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='asymptotic_generic', help='Report prefix; existing evidence is never overwritten')
    parser.add_argument('--cases', nargs='+', choices=('counting','automcstats','parametric'),
                        default=['counting','automcstats','parametric'])
    parser.add_argument('--modes', nargs='+', choices=('blind','expected','prefit'), default=['blind','expected','prefit'])
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    assert version('higgs-combine-tool') == '11.0.0'
    report = dict(status='running', script_sha256=sha(__file__), host=__import__('platform').node(),
        combine_version='11.0.0', relative_tolerance=.01, comparisons=[],
        sources={str(p):sha(p) for p in [*Path('src/combine_jax/methods').glob('*.py'),
            Path('src/combine_jax/tensor/limit_profile.py'),Path('src/combine_jax/cli.py')]},
        scope='Synthetic generic cards only. Native commands produce expected quantiles only. Sequential single runs; import/compile/execute reported separately. No HHVV data or observed limit.')
    save_json(args.output, report)
    for case in args.cases:
        source = Path('build/tensor_native_policy_inputs') / case
        workspace, card = source / 'workspace.root', source / 'card.txt'
        assert workspace.exists() and card.exists()
        for mode in args.modes:
            label = case + '_' + mode
            directory = args.directory / label
            directory.mkdir()
            options = ['--run', 'blind' if mode == 'blind' else 'expected']
            if mode == 'prefit':
                options += ['--noFitAsimov']
            native = [sys.executable, str(Path(sys.prefix)/'bin/combine'), '-M','AsymptoticLimits',
                      '-d',str(workspace.resolve()), *options, '-m','125','-n','.expected','-v','0']
            target = Path('reports') / (args.label + '_' + label + '.json')
            assert not target.exists()
            tensor = [sys.executable,'-m','combine_jax.cli','-M','AsymptoticLimits',
                '--workspace',str(workspace),'--datacard',str(card),'--poi','r',*options,
                '--output',str(target),'--workdir',str(directory/'prepared')]
            row = dict(case=case, mode=mode, input_workspace_sha256=sha(workspace),
                datacard_sha256=sha(card), native_command=native, tensor_command=tensor)
            for name, command, cwd, log in [('native',native,directory,directory/'combine.log'),
                    ('tensor',tensor,Path.cwd(),target.with_suffix('.log'))]:
                begin = time.perf_counter()
                with log.open('x') as stream:
                    result = subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
                row[name] = dict(returncode=result.returncode, wall_seconds=time.perf_counter()-begin,
                                 log=str(log),log_sha256=sha(log))
            if row['native']['returncode'] == 0 and row['tensor']['returncode'] == 0:
                from combine_jax.root_io import load_root
                root = load_root()
                assert root.gROOT.GetVersion() == '6.40.00'
                files = list(directory.glob('higgsCombine*.root'))
                assert len(files) == 1
                handle = root.TFile.Open(str(files[0]))
                values = sorted((float(r.quantileExpected),float(r.limit)) for r in handle.Get('limit'))
                handle.Close()
                assert len(values) == 5 and all(q>0 for q,x in values), 'Unexpected observed entry'
                measured = json.loads(target.read_text())
                assert measured['status'] == 'computed_pending_native_validation'
                assert all(r['status']==0 for r in measured['profile_fits'])
                assert all(abs(q-p)<1e-7 for (q,_),p in zip(values,measured['quantiles'],strict=True))
                errors = [abs(x-y)/abs(y) for x,(_,y) in zip(measured['limits'],values,strict=True)]
                row.update(status='passed' if max(errors)<.01 else 'failed_quantile_agreement',
                    native_file=str(files[0]),native_file_sha256=sha(files[0]), tensor_report=str(target),
                    tensor_report_sha256=sha(target), quantiles=measured['quantiles'],
                    native_limits=[v for _,v in values],tensor_limits=measured['limits'],relative_errors=errors,
                    tensor_timings=measured['timings'],tensor_preparation_seconds=measured['preparation']['preparation_seconds'],
                    tensor_peak_rss_mib=measured['peak_rss_mib'])
            else:
                row['status'] = 'failed_execution'
            report['comparisons'].append(row)
            save_json(args.output, report)
            print(label,row['status'],row.get('relative_errors'),flush=True)
    report['status'] = 'passed' if all(r['status']=='passed' for r in report['comparisons']) else 'failed'
    save_json(args.output, report)
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__ == '__main__':
    main()
