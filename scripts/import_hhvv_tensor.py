"""Fresh ROOT process: complete HHVV workspace+card to a neutral tensor artifact."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import time
import traceback

from combine_jax.root_io import open_workspace,load_caching_nll,REFERENCE_FLAGS
from combine_jax.tensor.model import from_workspace
from hhvv_setup import configure_hhvv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='build/tensor_hhvv_full_import')
    parser.add_argument('--output',default='reports/tensor_hhvv_full_import.json')
    parser.add_argument('--fit-policy',choices=('workspace','combine_v11_default'),default='workspace')
    args=parser.parse_args()
    path=Path(args.output);directory=Path(args.directory);assert not path.exists() and not directory.exists()
    directory.mkdir();started=time.perf_counter();root=load_caching_nll()
    census_path=Path('reports/structure_census.json');census=json.loads(census_path.read_bytes());source=census['workspace'];card=census['datacard']
    report=dict(status='running',stage='input_checks',script_sha256=sha(__file__),census=str(census_path),census_sha256=sha(census_path),
        source=source,card=dict(path=card['path'],sha256=card['sha256']),runtime_flags=REFERENCE_FLAGS,fit_policy=args.fit_policy,
        root_version=str(root.gROOT.GetVersion()),combine_version=version('higgs-combine-tool'),
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    try:
        assert sha(source['path'])==source['sha256'] and sha(card['path'])==card['sha256']
        report['stage']='import';save();before=time.perf_counter()
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors);root.RooAbsReal.clearEvalErrorLog()
        with open_workspace(source['path']) as workspace:
            assert configure_hhvv(workspace)==source['configuration']
            model=from_workspace(workspace,card['path'],fit_policy=args.fit_policy)
        report['import_seconds']=time.perf_counter()-before
        report['root_evaluation_errors']=int(root.RooAbsReal.numEvalErrors())
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.PrintErrors)
        assert report['root_evaluation_errors']==0 and len(model.metadata['channel_names'])==459
        assert model.metadata['provenance']['all_card_channels_imported']
        report['stage']='serialize';save();model.save(directory/'model')
        b=model.metadata['components']['binned'];grids=b['analytic_grid_queries']
        report.update(status='passed',stage='complete',model=str(directory/'model'),structure_key=model.structure_key,
            parameters=len(model.metadata['parameter_names']),channels=len(model.metadata['channel_names']),constraints=len(model.metadata['constraint_names']),
            profile_parameters=len(model.metadata['components']['templates']['profile_names']),
            buffer_bytes=sum(a.nbytes for a in model.arrays.values()),density_batches=len(b['program']),nll_batches=len(b['nll_program']),
            grid_queries=len(grids),folded_grid_queries=sum(q['folded'] for q in grids),
            grid_rows=sum(q['grid_rows'] for q in grids),uncompressed_grid_rows=sum(q['data_rows'] for q in grids),
            grid_expression_batches=sum(q['operation'].startswith('gridexpr_') for q in b['program']),
            grid_analytic_batches=sum(q['operation'].startswith('gridanalytic_') for q in b['program']),
            artifacts={str(p):sha(p) for p in (directory/'model').iterdir()})
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['import_seconds'],report['buffer_bytes'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
