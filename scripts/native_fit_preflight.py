"""Check the actual CLI start without running a minimizer."""
import subprocess
import time

from combine_jax.tensor.fit_policy import check_native_fit_state
from combine_jax.tensor.parameter_info import model_parameter_info
from tensor_benchmark_common import sha


def check_profile_bounds(snapshot, model):
    """Analytic bin nuisances are outside the explicit fitted parameter vector."""
    names=model.metadata['components'].get('templates',{}).get('profile_names',[])
    rows=[]
    for i,name in enumerate(names):
        var=snapshot.find(name);assert var,name
        bounds=(float(var.getMin()),float(var.getMax()))
        expected=(model.arrays['templates__profile_minimum'][i],model.arrays['templates__profile_maximum'][i])
        if bounds!=expected:
            raise ValueError(f'Actual CLI profiled nuisance bounds differ: {name}')
        rows.append(dict(name=name,lower=bounds[0],upper=bounds[1]))
    return rows


def run_preflight(command, directory, model, open_workspace, report, save):
    directory.mkdir()
    command = list(command)
    if '--saveFitResult' in command:
        command[command.index('--saveFitResult')] = '--saveWorkspace'
    command[command.index('-n')+1] = '.tensor_preflight'
    command += ['--skipInitialFit']
    log = directory/'combine.log'
    record = dict(status='running',command=command,script_sha256=sha(__file__),
        scope='Actual CLI with --algo none --skipInitialFit --saveWorkspace. No minimization; excluded from fit timing.')
    report['preflight'] = record;save()
    started=time.perf_counter()
    try:
        with log.open('x') as stream:
            completed=subprocess.run(command,cwd=directory,stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=True)
        record.update(returncode=completed.returncode,wall_seconds=time.perf_counter()-started,
            log=str(log),log_sha256=sha(log))
        text=log.read_text()
        if 'Doing initial fit:' in text or 'Minuit2Minimizer : Valid minimum' in text:
            raise ValueError('Preflight unexpectedly ran a minimizer')
        results=list(directory.glob('higgsCombine*.root'))
        assert len(results)==1
        record['workspace_sha256']=sha(results[0])
        with open_workspace(results[0]) as saved:
            assert saved.getSnapshot('clean') and saved.getSnapshot('MultiDimFit')
            saved.loadSnapshot('clean')
            record['initial_state']=check_native_fit_state(saved,model)
            # Snapshot MultiDimFit is taken before FitterAlgoBase restores bounds.
            snapshot=saved.getSnapshot('MultiDimFit')
            info=model_parameter_info(model)
            for i,name in enumerate(info.metadata['names']):
                var=snapshot.find(name);assert var,name
                value=float(var.getVal())
                bounds=(value,value) if var.isConstant() else (float(var.getMin()),float(var.getMax()))
                if value!=info.arrays['initial'][i] or bounds!=(info.arrays['lower'][i],info.arrays['upper'][i]):
                    raise ValueError(f'Actual CLI effective start differs: {name}')
            record['profiled_bounds']=check_profile_bounds(snapshot,model)
            data=saved.data(report['dataset'])
            record['dataset']=dict(name=report['dataset'],entries=int(data.numEntries()),sum_weights=float(data.sumEntries()))
            if 'dataset' in model.metadata['provenance']:
                from combine_jax.dataset_identity import workspace_dataset_identity
                record['dataset']['identity']=workspace_dataset_identity(saved,data)
                assert record['dataset']['identity']==model.metadata['provenance']['dataset']['identity']
        record['status']='passed'
    except BaseException:
        record['status']='failed'
        raise
    finally:
        record['wall_seconds']=time.perf_counter()-started
        save()
