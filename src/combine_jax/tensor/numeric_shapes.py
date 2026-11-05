"""Batched native numerical normalizations, with neutral formulas and buffers.

Only formula structure and integration policy are static. Model literals,
parameters and geometry remain runtime arrays. Integrals stream one shape at
a time; they never build a channel by quadrature-row reverse-AD tape.
"""
import numpy as np

from ..graph import UnsupportedModelError
from ..romberg import RombergConfig, integrate_numpy
from .scalar import operation, UNARY, BINARY


NUMERIC_CLASSES=frozenset(('RooGenericPdf','RooModZPdf'))
NUMERIC_OPERATIONS=frozenset(('numeric_generic_integral','numeric_modz_integral','numeric_modz_value','numeric_linear_basis_integral','numeric_product_basis_integral','numeric_joint_integral'))
_BINARY={'+':'add','-':'sub','*':'mul','/':'div','**':'pow','<':'lt','<=':'le','>':'gt','>=':'ge','==':'eq','!=':'ne'}


def bind_literals(tree, literals):
    """Move formula numbers into runtime buffers; preserve the allowlisted AST."""
    kind=tree[0]
    if kind=='constant':
        index=len(literals);literals.append(tree[1]);return ('literal',index)
    if kind=='argument':return tuple(tree)
    if kind in ('call','boolean'):
        return (kind,tree[1],tuple(bind_literals(t,literals) for t in tree[2]))
    if kind in ('binary','compare','unary'):
        return (kind,tree[1],*(bind_literals(t,literals) for t in tree[2:]))
    if kind=='conditional':return (kind,*(bind_literals(t,literals) for t in tree[1:]))
    raise UnsupportedModelError(f'Unknown numerical expression node {kind}')


def expression_value(tree, arguments, literals, xp):
    kind=tree[0]
    if kind=='argument':return arguments[tree[1]]
    if kind=='literal':return literals[tree[1]]
    if kind=='conditional':
        condition=expression_value(tree[1],arguments,literals,xp)
        def branch(t,a):
            shape=xp.broadcast_shapes(*(xp.shape(v) for v in a))
            return xp.broadcast_to(xp.asarray(expression_value(t,a,literals,xp),dtype=xp.float64),shape)
        if xp is np:
            # Select only active scalar branches, matching the JAX lax.map path.
            if np.ndim(condition)==0:return branch(tree[2] if condition else tree[3],arguments)
            arrays=np.broadcast_arrays(*arguments);shape=arrays[0].shape
            return np.array([expression_value(tree,tuple(a.flat[i] for a in arrays),literals,np)
                             for i in range(arrays[0].size)]).reshape(shape)
        import jax
        def scalar(a):
            return jax.lax.cond(xp.asarray(expression_value(tree[1],a,literals,xp),dtype=bool),
                lambda v:branch(tree[2],v),lambda v:branch(tree[3],v),a)
        if xp.ndim(condition)==0:return scalar(tuple(arguments))
        arrays=xp.broadcast_arrays(*arguments);shape=arrays[0].shape
        return jax.lax.map(scalar,tuple(a.reshape(-1) for a in arrays)).reshape(shape)
    if kind in ('binary','compare'):
        op=_BINARY[tree[1]];terms=tree[2:]
    elif kind=='unary':
        if tree[1]=='+':return expression_value(tree[2],arguments,literals,xp)
        op={'-':'neg','not':'not'}[tree[1]];terms=tree[2:]
    elif kind=='call':op={'fabs':'abs','pow':'pow'}.get(tree[1],tree[1]);terms=tree[2]
    elif kind=='boolean':
        terms=tree[2];result=expression_value(terms[0],arguments,literals,xp)
        for term in terms[1:]:
            other=expression_value(term,arguments,literals,xp)
            result=(xp.logical_and if tree[1]=='and' else xp.logical_or)(result,other)
        return result
    else:raise ValueError('Unknown numerical expression')
    operands=xp.stack(xp.broadcast_arrays(*(expression_value(t,arguments,literals,xp) for t in terms)),axis=-1)
    if xp is np:
        from scipy import special
    else:
        from jax.scipy import special
    return operation(op,operands,xp.ones(operands.shape,dtype=bool),xp,special)


def numeric_operation(op,scalars,b,spec,xp=np):
    params=xp.where(b['coef_mask'],scalars[b['coefs']],b['defaults'])
    if op in ('numeric_linear_basis_integral','numeric_product_basis_integral'):
        from ..romberg_table import integral, product_coefficients
        config=RombergConfig(**spec['integration_config'])
        coefficients=product_coefficients(params,spec['coefficient_groups'],xp) if op=='numeric_product_basis_integral' else xp.concatenate((xp.ones((len(params),1),dtype=xp.float64),params),axis=1)
        def compute(args):return integral(args[0],args[1],args[2][1]-args[2][0],config,xp)[0]
        if xp is np:result=np.array([compute(args) for args in zip(coefficients,b['refinement_table'],b['bounds'],strict=True)])
        else:
            import jax
            result=jax.lax.map(compute,(coefficients,b['refinement_table'],b['bounds']))
        return result[:,None]
    if op=='numeric_modz_value':
        from ..modz_math import value
        return value(b['data'],xp.swapaxes(params,0,1)[:,:,None],b['reference'][:,0,None],b['reference'][:,1,None],xp)
    config=RombergConfig(**spec['integration_config'])
    n_parameters=params.shape[1]
    def integrand(p,x):
        if op=='numeric_generic_integral':
            args=list(p[:n_parameters]);args.insert(spec['axis_index'],x)
            value=expression_value(spec['expression'],args,p[n_parameters:],xp)
        elif op=='numeric_joint_integral':
            from ..joint_product_math import value as joint_value
            value=joint_value(x,p[:n_parameters],spec['shape'],spec['shape_parameter_count'],p[n_parameters:].reshape(2,2),xp)
        else:
            from ..modz_math import value as modz_value
            value=modz_value(x,p[:n_parameters],p[-2],p[-1],xp)
        return xp.broadcast_to(value,xp.shape(x))
    extra=(b['literals'] if op=='numeric_generic_integral' else b['references'].reshape(len(params),4)
           if op=='numeric_joint_integral' else b['reference'])
    parameters=xp.concatenate((params,extra),axis=1)
    if xp is np:
        result=np.array([integrate_numpy(integrand,p,low,high,config)[0]
            for p,(low,high) in zip(parameters,b['bounds'],strict=True)])
    else:
        import jax
        from ..romberg_jax import integrate
        result=jax.lax.map(lambda args:integrate(integrand,args[0],args[1][0],args[1][1],config),
                           (parameters,b['bounds']))
    return result[:,None]


def validate_numeric_fields(op,b,spec,rows):
    if op not in NUMERIC_OPERATIONS or not isinstance(spec,dict):raise ValueError('Invalid numerical shape specification')
    size=len(b['outputs']);count=b['coefs'].shape[1]
    if b['defaults'].shape!=(size,count) or spec.get('parameter_count')!=count:
        raise ValueError('Invalid numerical defaults or parameter arity')
    if op.endswith('_integral'):
        RombergConfig(**spec['integration_config'])
        if rows!=1 or b['bounds'].shape!=(size,2) or np.any(b['bounds'][:,1]<=b['bounds'][:,0]):
            raise ValueError('Invalid numerical integration bounds')
    if op=='numeric_linear_basis_integral':
        if b['refinement_table'].shape!=(size,spec['integration_config']['max_steps'],count+1):
            raise ValueError('Invalid linear integration basis table')
    elif op=='numeric_product_basis_integral':
        groups=spec.get('coefficient_groups')
        if not isinstance(groups,(tuple,list)) or any(type(n) is not int or n<0 for n in groups) or sum(groups)!=count:
            raise ValueError('Invalid polynomial coefficient groups')
        if b['refinement_table'].shape!=(size,spec['integration_config']['max_steps'],int(np.prod([n+1 for n in groups]))):
            raise ValueError('Invalid product integration basis table')
    elif op=='numeric_generic_integral':
        if b['literals'].ndim!=2 or len(b['literals'])!=size or not 0<=spec['axis_index']<=count:
            raise ValueError('Invalid numerical formula arguments')
        def check(tree):
            kind=tree[0]
            if kind in ('argument','literal'):
                limit=count+1 if kind=='argument' else b['literals'].shape[1]
                if type(tree[1]) is not int or not 0<=tree[1]<limit:raise ValueError('Invalid numerical formula index')
                return
            if kind in ('binary','compare'):
                if tree[1] not in _BINARY or len(tree)!=4:raise ValueError('Invalid numerical binary expression')
                terms=tree[2:]
            elif kind=='unary':
                if tree[1] not in ('+','-','not') or len(tree)!=3:raise ValueError('Invalid numerical unary expression')
                terms=tree[2:]
            elif kind=='call':
                name={'fabs':'abs'}.get(tree[1],tree[1]);arity=1 if name in UNARY else 2 if name in BINARY else 3 if name=='landau' else None
                if arity is None or len(tree[2])!=arity:raise ValueError('Invalid numerical function')
                terms=tree[2]
            elif kind=='boolean':
                if tree[1] not in ('and','or') or not tree[2]:raise ValueError('Invalid numerical boolean expression')
                terms=tree[2]
            elif kind=='conditional':
                if len(tree)!=4:raise ValueError('Invalid numerical conditional expression')
                terms=tree[1:]
            else:raise ValueError('Unknown numerical expression node')
            for term in terms:check(term)
        check(spec['expression'])
    elif op=='numeric_joint_integral':
        n=spec.get('shape_parameter_count')
        if (type(n) is not int or n>count or spec.get('shape') not in ('modz','twoexp')
                or (n!=4 if spec['shape']=='twoexp' else n<5)
                or b['references'].shape!=(size,2,2)
                or np.any(b['references'][:,:,1]<=b['references'][:,:,0])):
            raise ValueError('Invalid joint numerical shape geometry')
    else:
        if count<5 or b['reference'].shape!=(size,2) or np.any(b['reference'][:,1]<=b['reference'][:,0]):
            raise ValueError('Invalid ModZ geometry')
        if op.endswith('_value') and b['data'].shape!=(size,rows):raise ValueError('Invalid ModZ coordinates')
