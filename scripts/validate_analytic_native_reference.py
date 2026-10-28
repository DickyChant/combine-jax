"""Existing parametric-card harness in an isolated Combine v11 process."""
import argparse
import hashlib
import json
from pathlib import Path

from combine_jax.root_io import load_caching_nll,open_workspace
from combine_jax.validation import validate_workspace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('workspace');parser.add_argument('output');args=parser.parse_args()
    output=Path(args.output);assert not output.exists()
    root=load_caching_nll();root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    with open_workspace(args.workspace) as workspace:
        root.RooAbsReal.clearEvalErrorLog();report=validate_workspace(workspace,random_points=25,seed=1820)
        report['root_evaluation_errors']=int(root.RooAbsReal.numEvalErrors())
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.PrintErrors)
    passed=(report['max_nll_absolute_error']<1e-6 and report['max_gradient_relative_error']<1e-4
        and report['max_hessian_scaled_error']<1e-6 and report['root_evaluation_errors']==0)
    report.update(status='passed' if passed else 'failed',workspace_sha256=hashlib.sha256(Path(args.workspace).read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print({k:report[k] for k in ('status','max_nll_absolute_error','max_gradient_relative_error',
        'max_hessian_absolute_error','max_hessian_scaled_error','root_evaluation_errors')},flush=True)
    assert passed


if __name__=='__main__':main()
