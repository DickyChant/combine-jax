"""Sequential same-node JIT/eager/CPU/GPU/native runs; each operation is isolated."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--cases', nargs='+', default=['counting', 'automcstats'])
    parser.add_argument('--backends', nargs='+', choices=['cpu','gpu'], default=['cpu','gpu'])
    parser.add_argument('--modes', nargs='+', choices=['jit','eager'], default=['jit'])
    parser.add_argument('--operations', nargs='+', choices=['nll','value_gradient','hessian','fit'], default=['nll','value_gradient'])
    parser.add_argument('--repeat', type=int, default=26)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    assert args.tag.replace('_','').isalnum() and args.repeat > 0
    manifest = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    output = Path('reports')/('gpu_tensor_'+args.tag+'_driver.json')
    assert not output.exists()
    output.parent.mkdir(exist_ok=True)
    report = dict(status='running', started_unix=time.time(), commands=[], settings=vars(args))
    def save():
        tmp = output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(output)
    def run(command, log):
        print('run', command, flush=True)
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save()
        with log.open('x') as stream:
            child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row.update(status='passed' if child.returncode == 0 else 'failed', returncode=child.returncode,
                   finished_unix=time.time()); save()
        child.check_returncode()
    save()
    try:
        if 'gpu' in args.backends:
            probe = Path('reports')/f'gpu_tensor_{args.tag}_environment.json'
            run(['bash','scripts/run_tensor_lcg.sh','scripts/verify_tensor_gpu.py','--output',str(probe)], probe.with_suffix('.log'))
        for label in args.cases:
            case = manifest[label]
            measured = []
            for backend in args.backends:
                for mode in args.modes:
                    for operation in args.operations:
                        tag = f'gpu_tensor_{args.tag}_{label}_{backend}_{mode}_{operation}'
                        path = Path('reports')/(tag+'.json')
                        command = ['bash','scripts/run_tensor_lcg.sh','scripts/benchmark_tensor.py',
                            '--model',case['model'],'--points',case['curvature_points'] if operation == 'hessian' else case['points'],
                            '--backend',backend,'--mode',mode,'--operation',operation,'--repeat',str(args.repeat),
                            '--output',str(path)]
                        run(command, path.with_suffix('.log'))
                        measured.append(str(path))
            audit = Path('reports')/f'gpu_tensor_{args.tag}_{label}_audit.json'
            run(['bash','scripts/run_tensor_lcg.sh','scripts/audit_tensor_benchmarks.py','--case',label,
                 '--reports',*measured,'--output',str(audit)], audit.with_suffix('.log'))
            if args.native:
                path = Path('reports')/f'gpu_tensor_{args.tag}_{label}_native.json'
                run(['bash','scripts/run_tensor_native.sh','scripts/benchmark_native_tensor.py','--case',label,
                    '--repeat',str(args.repeat),'--output',str(path)], path.with_suffix('.log'))
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['finished_unix'] = time.time(); save()


if __name__ == '__main__':
    main()
