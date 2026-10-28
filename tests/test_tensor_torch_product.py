"""Product reductions keep polynomial derivatives at zeros without host reads."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from combine_jax.tensor.torch_backend import _ArrayAPI


@pytest.mark.parametrize('shape,axis', [((3,5),-1),((2,3,4),1),((2,3),None),((3,0),1),((1,3),0),((),None)])
@pytest.mark.parametrize('keepdims',[False,True])
def test_product_shapes_and_empty_identity(shape,axis,keepdims):
    values=np.linspace(-1.7,2.1,max(1,int(np.prod(shape))))
    values=values[:int(np.prod(shape))].reshape(shape)
    x=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    actual=_ArrayAPI(torch).prod(x,axis=axis,keepdims=keepdims)
    expected=np.prod(values,axis=axis,keepdims=keepdims)
    assert actual.shape==expected.shape
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=1e-14,rtol=1e-14)
    gradient=torch.autograd.grad(actual.sum(),x)[0]
    assert gradient.shape==x.shape and torch.isfinite(gradient).all()


@pytest.mark.parametrize('values',[[0.,2.,3.],[0.,0.,3.],[0.,0.,0.],[-2.,.5,-3.],[1.,-2.,0.,3.,.2]])
def test_zero_and_signed_product_gradient_hessian(values):
    x=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    fn=lambda t:_ArrayAPI(torch).prod(t)
    assert torch.autograd.gradcheck(fn,(x,))
    assert torch.autograd.gradgradcheck(fn,(x,))
    hessian=torch.autograd.functional.hessian(fn,x).numpy()
    expected=np.zeros((len(values),len(values)))
    for i in range(len(values)):
        for j in range(len(values)):
            if i!=j:expected[i,j]=np.prod([v for k,v in enumerate(values) if k not in (i,j)])
    np.testing.assert_allclose(hessian,expected,atol=1e-14,rtol=1e-14)
