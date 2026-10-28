"""Build and audit call instrumentation against the active native ROOT headers."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import sha


def build_library(directory, root):
    source=Path(__file__).with_suffix('.cc').resolve()
    package=Path(root.__file__).resolve().parent
    include=package/'include'; libraries=package/'lib'
    output=(directory/'native_minuit_counters.so').resolve()
    command=['g++','-std=c++17','-O2','-fPIC','-shared',str(source),'-I',str(include),
             '-L',str(libraries),'-Wl,-rpath,'+str(libraries),'-lMinuit2','-ldl','-o',str(output)]
    started=time.perf_counter()
    process=subprocess.run(command,capture_output=True,text=True)
    log=directory/'counter_build.log';log.write_text(process.stdout+process.stderr)
    assert process.returncode==0,log
    return output,dict(source=str(source),source_sha256=sha(source),library=str(output),library_sha256=sha(output),
        root_version=root.gROOT.GetVersion(),include_directory=str(include),library_directory=str(libraries),
        command=command,compiler_version=subprocess.check_output(['g++','--version'],text=True),
        build_seconds=time.perf_counter()-started,build_log=str(log),build_log_sha256=sha(log),
        headers={str(p.relative_to(include)):sha(p) for p in (include/'Minuit2').glob('*.h')},
        scope='LD_PRELOAD wrappers forward the original installed symbols unchanged. Counters include nested-call suppression for numerical gradient entry points. Library build is outside the fit timer.')


def read_counts(prefix, native_function_calls):
    files=sorted(prefix.parent.glob(prefix.name+'.*.jsonl'))
    events=[]
    for path in files:
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    minimize=[r for r in events if r['event']=='minimize_return']
    assert len(minimize)==1 and minimize[0]['success'],events
    current=minimize[0]
    assert current['phase_nll']>0 and current['phase_numerical_gradients']>0,current
    assert not any(r['event']=='minimize_exception' for r in events)
    assert any(r['event']=='process_exit' and r['pid']==current['pid'] for r in events)
    return dict(files={str(p):sha(p) for p in files},events=events),dict(
        function=native_function_calls,actual_nll_evaluations=current['phase_nll'],analytic_gradient_callbacks=0,
        numerical_gradient_evaluations=current['phase_numerical_gradients']+current['phase_hessian_gradients'],
        numerical_gradient_iterations=current['phase_numerical_gradients'],
        hessian_gradient_checks=current['phase_hessian_gradients'],
        scope='Minuit2 Nfcn counts objective callbacks; actual_nll_evaluations counts CachingSimNLL::evaluate inside Minimize. Numerical gradients count top-level calculator invocations, including Hessian gradient checks and excluding nested forwarding calls. Both derivative kinds already contribute to native Nfcn. Startup and canonical postchecks are outside this phase.')
