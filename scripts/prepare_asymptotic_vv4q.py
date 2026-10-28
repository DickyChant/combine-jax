"""Stage the twelve VV4q cards and build v11 workspaces without running fits."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physics-repository', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('full','noVH','onlyVH'), default=['full','noVH','onlyVH'])
    parser.add_argument('--masses', nargs='+', type=int, default=[1300,2000,3000,4000])
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    assert version('higgs-combine-tool') == '11.0.0'
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    repo = args.physics_repository.resolve()
    # Keep Python bytecode, law configuration and all generated files in this
    # project. Import the actual physics model; no dependency or rate shims.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', DHI_COMBINE_VERSION='v11.0.0',
        LAW_HOME=str((args.directory/'law').resolve()), LAW_CONFIG_FILE='/dev/null',
        DHI_HHVV_INPUTS=str(repo/'dhi/models/hhvv_inputs'), DHI_HAS_GFAL='0', DHI_USE_GFAL='0',
        PYTHONPATH=os.pathsep.join(map(str,[repo,repo/'dhi/models',repo/'modules/law'])))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        env[name] = '1'
    physics_files = [repo/p for p in ['dhi/__init__.py','dhi/config.py','dhi/util.py',
        'dhi/models/hhvv_model.py','dhi/models/hhvv_inputs/XS_ggF.csv','dhi/models/hhvv_inputs/BR.csv']]
    report = dict(status='running', script_sha256=sha(__file__), inventory_sha256=sha(args.inventory),
        combine_version=version('higgs-combine-tool'), physics_model='hhvv_model.model_default',
        physics_repository=str(repo), physics_sources={str(p):sha(p) for p in physics_files},
        dependencies={name:version(name) for name in ('luigi','scinum','pandas','scipy','root')},
        environment={k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','DHI_COMBINE_VERSION','LAW_HOME',
            'LAW_CONFIG_FILE','DHI_HHVV_INPUTS','DHI_HAS_GFAL','DHI_USE_GFAL','PYTHONPATH')},
        scope='Translation only, no fits. Source cards and ROOT originals remain unchanged. Independent dhi expected-limit references are pending.',
        shape_files={}, workspaces=[])
    report['law_revision'] = subprocess.check_output(['git','-C',str(repo/'modules/law'),'rev-parse','HEAD'],text=True).strip()
    report['physics_revision'] = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    save_json(args.output, report)
    cache = args.directory/'shape_files'
    cache.mkdir()
    for row in json.loads(args.inventory.read_text())['inputs']:
        if row['variant'] not in args.variants or row['mass'] not in args.masses:
            continue
        directory = args.directory/f"{row['variant']}_{row['mass']}"
        directory.mkdir()
        card = directory/'card.txt'
        assert sha(row['card']) == row['card_sha256']
        shutil.copyfile(row['card'], card)
        for name in row['shape_files']:
            if name not in report['shape_files']:
                checksum = sha(name)
                saved = cache/(checksum+'.root')
                if not saved.exists():
                    shutil.copyfile(name, saved)
                    assert sha(saved) == checksum
                report['shape_files'][name] = dict(sha256=checksum, path=str(saved), bytes=saved.stat().st_size)
            record = report['shape_files'][name]
            os.link(record['path'], directory/Path(name).name)
        command = [sys.executable, str(Path(sys.prefix)/'bin/text2workspace.py'), str(card.resolve()),
            '-P','hhvv_model:model_default','-m',str(row['mass']),'-o',str((directory/'workspace.root').resolve())]
        begin = time.perf_counter()
        with (directory/'text2workspace.log').open('x') as stream:
            result = subprocess.run(command,env=env,cwd=directory,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        record = dict(variant=row['variant'], mass=row['mass'], channels=row['channels'],
            source_card_sha256=row['card_sha256'], card=str(card), command=command,
            returncode=result.returncode, wall_seconds=time.perf_counter()-begin,
            log=str(directory/'text2workspace.log'), log_sha256=sha(directory/'text2workspace.log'))
        if result.returncode == 0:
            workspace = directory/'workspace.root'
            record.update(workspace=str(workspace), workspace_sha256=sha(workspace))
            from combine_jax.root_io import load_root, open_workspace
            root = load_root()
            assert root.gROOT.GetVersion() == '6.40.00'
            with open_workspace(workspace) as ws:
                mx = ws.var('MX')
                assert mx and mx.isConstant() and mx.getVal() == row['mass']
                pois = [str(v.GetName()) for v in ws.obj('ModelConfig').GetParametersOfInterest()]
                assert pois == ['r_hhvv']
                record['configuration'] = dict(MX=float(mx.getVal()), MX_frozen=bool(mx.isConstant()), pois=pois)
        report['workspaces'].append(record)
        save_json(args.output,report)
        print(row['variant'],row['mass'],'built' if result.returncode == 0 else 'failed',flush=True)
        if result.returncode:
            report['status'] = 'failed_translation'
            save_json(args.output,report)
            raise RuntimeError(f'Translation failed; see {record["log"]}')
    # Recheck every staged original after the translator has used it.
    assert all(sha(v['path']) == v['sha256'] for v in report['shape_files'].values())
    report['status'] = 'workspaces_built_no_fits'
    save_json(args.output,report)


if __name__ == '__main__':
    main()
