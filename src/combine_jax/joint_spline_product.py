"""Native spline times polynomial products, including nested RooProdPdf."""
import numpy as np

from .graph import UnsupportedModelError


def extract(root, specifications):
    if any(kind!='all' for kind,_ in specifications):return None
    products=[]
    def leaves(pdf):
        if pdf.ClassName()!='RooProdPdf':
            return ([pdf] if pdf.ClassName() in ('RooGenericPdf','RooChebychev')
                    or pdf.ClassName().startswith('RooBernsteinFast<') else None)
        products.append(pdf);result=[]
        for child in pdf.pdfList():
            specified=pdf.findPdfNSet(child)
            if specified and len(specified):return None
            nested=leaves(child)
            if nested is None:return None
            result.extend(nested)
        return result
    children=leaves(root)
    if children is None:return None
    wrappers=[c for c in children if c.ClassName()=='RooGenericPdf']
    polynomials=[c for c in children if c.ClassName()=='RooChebychev' or c.ClassName().startswith('RooBernsteinFast<')]
    if len(wrappers)!=1 or not polynomials or len(polynomials)+1!=len(children):return None
    wrapper=wrappers[0]
    dependencies=list(wrapper.dependents())
    if len(dependencies)!=1 or dependencies[0].ClassName()!='RooSpline1D':return None
    from .formula import parse_formula
    if parse_formula(str(wrapper.expression()),[str(dependencies[0].GetName())])!=('argument',0):return None
    from .splines import extract_spline
    from .spline_math import value
    from .chebyshev import extract_chebyshev
    from .bernstein_fast import extract_fast_bernstein
    from .chebyshev_geometry import basis
    from .romberg import extract_config, RombergConfig
    from .romberg_table import build_table
    spline_inputs,spline=extract_spline(dependencies[0])
    axis=next(iter(dependencies[0].servers()))
    if axis.ClassName()!='RooRealVar':return None
    shapes=[];names=[];groups=[]
    for polynomial in polynomials:
        cheby=polynomial.ClassName()=='RooChebychev'
        _,attrs=(extract_chebyshev if cheby else extract_fast_bernstein)(polynomial)
        if spline_inputs!=(attrs['observable_name'],):return None
        servers={str(c.GetName()):c for c in polynomial.servers()}
        if any(servers[n].dependsOn(axis) for n in attrs['coefs']):
            raise UnsupportedModelError('Joint spline/polynomial coefficients depend on the integration axis')
        shapes.append((cheby,attrs));names.extend(attrs['coefs']);groups.append(len(attrs['coefs']))
    low,high=float(axis.getMin()),float(axis.getMax())
    if low<spline['knots'][0] or high>spline['knots'][-1]:
        raise UnsupportedModelError('Joint spline integration range is outside its stored domain')
    config=extract_config(root)
    # RooProdPdf's generated denominator uses its default integration policy;
    # custom per-object policies need a separate native validation.
    from dataclasses import asdict
    if config!=asdict(RombergConfig()) or any(config!=extract_config(p) for p in [wrapper,*products,*polynomials]):
        raise UnsupportedModelError('Joint product custom numerical configuration is not yet certified')
    def functions(x):
        result=value(x,spline,np)[None,:]
        for cheby,attrs in shapes:
            degree=len(attrs['coefs'])
            a,b=attrs['reference_range'] if cheby else attrs['range']
            if cheby:
                t=(x-.5*(a+b))/(.5*(b-a))
                columns=np.concatenate((np.ones((1,len(x))),basis(degree,t)),axis=0)
            else:
                t=(x-a)/(b-a)
                columns=np.asarray(attrs['matrix']).T@np.stack([t**i for i in range(degree+1)])
            result=(result[:,None,:]*columns[None,:,:]).reshape(-1,len(x))
        return result
    table=build_table(functions,low,high,RombergConfig(**config))
    return dict(axis=str(axis.GetName()),coefs=tuple(names),bounds=(low,high),coefficient_groups=tuple(groups),
        unit_integral_factors=tuple(str(c.GetName()) for c in root.pdfList() if c.ClassName()=='RooProdPdf'),
        integration_config=config,refinement_table=table,
        raw_integral_convention='native component integrals; nested self-normalized products contribute one',
        density_convention='joint raw-product numerical denominator')


def density(node,context,axes):
    import jax.numpy as jnp
    from .romberg import RombergConfig
    from .romberg_table import integral
    a=node.attrs['joint_spline_cheby']
    if axes!=(a['axis'],) or context.bounds(a['axis'])!=tuple(a['bounds']):
        raise UnsupportedModelError(f'{node.name}: joint spline product requires its exported normalization axis/range')
    from .romberg_table import product_coefficients
    coefficients=jnp.stack(context.values(a['coefs'])) if a['coefs'] else jnp.empty(0,dtype=jnp.float64)
    if coefficients.ndim!=1:raise UnsupportedModelError('Conditional joint spline coefficients are not translated')
    parameters=product_coefficients(coefficients,a.get('coefficient_groups',(len(a['coefs']),)),jnp)
    denominator=integral(parameters,jnp.asarray(a['refinement_table']),a['bounds'][1]-a['bounds'][0],
                         RombergConfig(**a['integration_config']),jnp)[0]
    return context.value(node.name)/denominator
