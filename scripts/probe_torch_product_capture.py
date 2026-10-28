"""Isolate product-backward CUDA capture and verify replayed polynomial gradients."""
import argparse
import os
from pathlib import Path
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('builtin','adapter'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    os.environ['TORCH_SHOW_CPP_STACKTRACES']='1'
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    import numpy as np
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    x=torch.tensor([[1.,2.,3.,4.,5.],[0.,-2.,3.,1.,.5],[0.,0.,-2.,1.,3.]],device='cuda',dtype=torch.float64,requires_grad=True)
    assert x.device.type=='cuda' and x.dtype==torch.float64
    monitor=Resources(True,args.output)
    report=dict(provenance(),status='checking',mode=args.mode,script_sha256=sha(__file__),
        torch_version=torch.__version__,torch_cuda_version=torch.version.cuda,float64=True,
        gpu=torch.cuda.get_device_name(),cases=[],
        upstream_source='https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp#L764-L790')
    def save():
        report['resources']=monitor.snapshot();save_json(args.output,report)
    save()
    from combine_jax.tensor.torch_backend import _ArrayAPI
    product=(lambda v:torch.prod(v,dim=-1)) if args.mode=='builtin' else (lambda v:_ArrayAPI(torch).prod(v,axis=-1))
    def invoke():
        value=product(x).sum()
        return value,torch.autograd.grad(value,x)[0]
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):warm=invoke()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();del warm
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):output=invoke()
        except BaseException:
            report.update(status='reproduced_capture_failure' if args.mode=='builtin' else 'failed',exception=traceback.format_exc())
            save();print(report['exception'],flush=True)
            if args.mode!='builtin':raise
            return
        for values in (np.array([[1.,2.,3.,4.,5.],[0.,-2.,3.,1.,.5],[0.,0.,-2.,1.,3.]]),
                       np.array([[-1.,.5,3.,2.,1.],[1.,0.,3.,-2.,4.],[2.,.5,1.,3.,-.5]])):
            with torch.no_grad():x.copy_(torch.tensor(values,device='cuda',dtype=torch.float64))
            graph.replay();torch.cuda.synchronize()
            value=float(output[0].detach().cpu());gradient=output[1].detach().cpu().numpy().copy()
            expected_value=np.prod(values,axis=-1).sum()
            expected_gradient=np.stack([np.prod(np.delete(values,i,axis=1),axis=1) for i in range(values.shape[1])],axis=1)
            np.testing.assert_allclose(value,expected_value,atol=1e-13,rtol=1e-13)
            np.testing.assert_allclose(gradient,expected_gradient,atol=1e-13,rtol=1e-13)
            report['cases'].append(dict(input=values,value=value,gradient=gradient,
                value_error=abs(value-expected_value),gradient_max_error=float(np.max(np.abs(gradient-expected_gradient)))))
        report['status']='passed_replayed_polynomial_gradients';save();print(report['status'],flush=True)
    finally:
        monitor.close();save()


if __name__=='__main__':
    main()
