"""A device fit loop followed by exact AD curvature and its unmodified inverse."""
from .optimize import make_bfgs
from .curvature import make_value_gradient_hessian
from .reported_errors import minuit_reported_errors
from .parameter_info import COORDINATE_FIELDS


COVARIANCE_STATUS = {0: 'positive_definite', 1: 'fit_not_converged',
                     2: 'nonfinite_hessian', 3: 'asymmetric_hessian',
                     4: 'nonpositive_curvature', 5: 'inverse_failed',
                     6: 'not_requested', 7: 'all_fixed'}


def physical_parameters(z, coordinates):
    """Map feasible coordinates to physical bounds with affine AD derivatives.

    Fused origin + z*scale can round outside an exact physical bound, even when
    z equals its scaled bound. Correct that rounding in the primal value while
    retaining the affine derivative for the external NLL gradient/Hessian.
    Bounds come directly from metadata, never from an inverse scale round trip.
    """
    import jax
    import jax.numpy as jnp
    c = coordinates
    raw = c['reference'][c['free_indices']]+z*c['scale']
    bounded = jnp.clip(raw, c['physical_lower'], c['physical_upper'])
    corrected = raw+jax.lax.stop_gradient(bounded-raw)
    return c['reference'].at[c['free_indices']].set(corrected, unique_indices=True)


def make_fit_function(function, *, covariance=True, column_batch_size=4,
                      curvature_linearization='reuse',
                      max_iterations=1000, max_linesearch=40, gradient_tolerance=1e-6,armijo_roundoff_multiplier=8.,
                      stopping_criterion='projected_gradient', edm_tolerance=1e-4, max_edm_refinements=3,
                      edm_validation='exact_hessian'):
    """Return ``fit(coordinates, model_arrays)`` for one outer JIT compilation.

    Coordinates are numeric runtime fields produced by ``fit_coordinates``:
    reference, unique free_indices, positive scale, scaled lower/upper and exact
    physical_lower/physical_upper. Host setup
    validates these before dispatch; no parameter or intermediate state is copied
    to the host during fitting. The covariance is the raw inverse of the exact
    free-coordinate Hessian, transformed to physical units. At an active bound
    it is a local curvature covariance, not a bounded confidence interval.
    Invalid covariance matrices are NaN with an explicit status, never repaired.
    EDM candidates with valid positive exact curvature are checked against the
    same raw EDM threshold and refined on device using that inverse as the next
    search metric. At most max_edm_refinements refinements share the original
    iteration budget. Invalid curvature remains explicit and cannot certify EDM;
    it does not erase the optimizer's separately reported estimated-EDM status.
    Explicit ``edm_validation='estimated'`` retains exact final curvature as a
    diagnostic but disables curvature-based refinement. This is intended for
    separately validated nonsmooth objectives; it does not certify a minimum
    or covariance and never changes the estimated-EDM threshold.
    With ``covariance=False`` and estimated EDM, fitting uses only values and
    gradients. No final Hessian is evaluated; covariance/error fields remain
    explicitly unavailable. Separate HVP uncertainty calculations can follow.
    ``errors`` and ``curvature_errors`` are covariance diagonal square roots;
    ``reported_errors`` separately applies Minuit's bounded finite-step convention.
    ``curvature_linearization='recompute'`` rematerializes gradient intermediates
    for each Hessian column batch instead of retaining the linearization tape.
    It preserves exact AD curvature and all convergence thresholds.
    """
    import jax
    import jax.numpy as jnp
    if type(covariance) is not bool:raise ValueError('covariance must be boolean')
    if type(max_edm_refinements) is not int or max_edm_refinements<0:
        raise ValueError('max_edm_refinements must be a nonnegative integer')
    if edm_validation not in ('exact_hessian','estimated'):
        raise ValueError('edm_validation must be exact_hessian or estimated')

    def objective(z, data):return function(physical_parameters(z,data['coordinates']),data['model'])
    optimize=make_bfgs(objective,max_iterations=max_iterations,max_linesearch=max_linesearch,
        gradient_tolerance=gradient_tolerance,armijo_roundoff_multiplier=armijo_roundoff_multiplier,
        stopping_criterion=stopping_criterion,edm_tolerance=edm_tolerance)
    curvature=make_value_gradient_hessian(objective,column_batch_size=column_batch_size,
        linearization=curvature_linearization)

    def fit(coordinates,arrays):
        c=coordinates;n=c['free_indices'].shape[0];total=c['reference'].shape[0]
        data=dict(coordinates=c,model=arrays)
        result=optimize(jnp.zeros(n,dtype=jnp.float64),data,c['lower'],c['upper'])
        if n and (covariance or (stopping_criterion=='edm' and edm_validation=='exact_hessian')):
            def evaluate(state):
                _,_,h=curvature(state['x'],data)
                finite=jnp.all(jnp.isfinite(h))
                asymmetry=jnp.max(jnp.abs(h-h.T))/jnp.maximum(1.,jnp.max(jnp.abs(h)))
                symmetric=asymmetry<=1e-8
                cholesky=jnp.linalg.cholesky(h,symmetrize_input=False)
                positive=jnp.all(jnp.isfinite(cholesky))&jnp.all(jnp.diag(cholesky)>0)
                inverse=jnp.linalg.inv(h)
                residual=jnp.max(jnp.abs(h@inverse-jnp.eye(n,dtype=h.dtype)))
                valid_inverse=jnp.all(jnp.isfinite(inverse))&(residual<=1e-6)&jnp.all(jnp.diag(inverse)>0)
                code=jnp.where(~finite,2,jnp.where(~symmetric,3,jnp.where(~positive,4,jnp.where(~valid_inverse,5,0))))
                # EDM concerns feasible motion: bound-blocked coordinates are
                # fixed in this search metric. The reported covariance remains
                # the unmodified inverse of the full physical Hessian.
                eps=32*jnp.finfo(jnp.float64).eps*(1+jnp.abs(state['x']))
                blocked=((state['x']<=c['lower']+eps)&(state['gradient']>0))|((state['x']>=c['upper']-eps)&(state['gradient']<0))
                reduced=jnp.where(blocked[:,None]|blocked[None,:],jnp.eye(n,dtype=h.dtype),h)
                metric=jnp.linalg.inv(reduced) if stopping_criterion=='edm' else inverse
                return h,inverse,code,asymmetry,residual,metric
            invalid=jnp.full((n,n),jnp.nan,dtype=jnp.float64)
            def check(state):
                return jax.lax.cond(state['status']==0,evaluate,
                    lambda _:(invalid,invalid,jnp.int64(1),jnp.float64(jnp.nan),jnp.float64(jnp.nan),invalid),state)
            checked=check(result)
            refinements=jnp.int32(0)
            hessian_evaluations=(result['status']==0).astype(jnp.int32)
            if stopping_criterion=='edm' and edm_validation=='exact_hessian':
                def exact_edm(state, matrix):
                    pg=state['projected_gradient']
                    return .5*jnp.dot(pg,matrix@pg)
                def needs_refinement(state, curvature_result):
                    edm=exact_edm(state,curvature_result[5])
                    return (state['status']==0)&(curvature_result[2]==0)&((edm>=edm_tolerance)|(edm<0)|~jnp.isfinite(edm))
                def condition(state):
                    current,matrices,count,calls=state
                    return (count<max_edm_refinements)&(current['iteration']<max_iterations)&needs_refinement(current,matrices)
                def body(state):
                    previous,matrices,count,calls=state
                    current=optimize(previous['x'],data,c['lower'],c['upper'],
                        initial_inverse=matrices[5],initial_inverse_uncertainty=0.,
                        iteration_budget=max_iterations-previous['iteration'])
                    for key in ('iteration','function_evaluations','gradient_evaluations','line_search_evaluations',
                                'resets','skipped_updates','roundoff_accepts'):
                        current[key]=current[key]+previous[key]
                    current['maximum_roundoff_allowance']=jnp.maximum(current['maximum_roundoff_allowance'],previous['maximum_roundoff_allowance'])
                    return current,check(current),count+1,calls+(current['status']==0).astype(jnp.int32)
                result,checked,refinements,hessian_evaluations=jax.lax.while_loop(condition,body,
                    (result,checked,refinements,hessian_evaluations))
                failed=needs_refinement(result,checked)
                result['status']=jnp.where(failed,5,result['status'])
            h,inv,code,asymmetry,residual,metric=checked
            result['hessian_edm']=.5*jnp.dot(result['projected_gradient'],metric@result['projected_gradient'])
            result['edm_curvature_validated']=(code==0)&jnp.isfinite(result['hessian_edm'])&(result['hessian_edm']>=0)&(result['hessian_edm']<edm_tolerance)
            result['edm_refinements']=refinements
            result['hessian_evaluations']=hessian_evaluations
            code=jnp.where(result['status']==0,code,1)
            result['hessian']=h/c['scale'][:,None]/c['scale'][None,:]
            physical_cov=inv*c['scale'][:,None]*c['scale'][None,:]
            full=jnp.zeros((total,total),dtype=jnp.float64).at[
                c['free_indices'][:,None],c['free_indices'][None,:]].set(physical_cov,unique_indices=True)
            result['covariance']=jnp.where(code==0,full,jnp.full_like(full,jnp.nan))
            result['covariance_status']=code if covariance else jnp.int64(6)
            if not covariance:result['covariance']=jnp.full_like(full,jnp.nan)
            result['hessian_relative_asymmetry']=asymmetry
            result['inverse_residual']=residual
        else:
            result['hessian']=jnp.full((n,n),jnp.nan,dtype=jnp.float64)
            result['covariance']=jnp.full((total,total),jnp.nan,dtype=jnp.float64) if n or not covariance else jnp.zeros((total,total),dtype=jnp.float64)
            result['covariance_status']=jnp.int64(6) if not covariance else jnp.where(result['status']==0,7,1)
            if covariance:result['covariance']=jnp.where(result['status']==0,result['covariance'],jnp.nan)
            result['hessian_relative_asymmetry']=jnp.float64(jnp.nan)
            result['inverse_residual']=jnp.float64(jnp.nan)
        result['values']=physical_parameters(result['x'],c)
        raw=c['reference'][c['free_indices']]+result['x']*c['scale']
        result['final_coordinate_roundoff_correction']=result['values'][c['free_indices']]-raw
        result['free_gradient']=result['gradient']/c['scale']
        eps=32*jnp.finfo(jnp.float64).eps*(1+jnp.abs(result['x']))
        result['active_bounds']=(result['x']<=c['lower']+eps)|(result['x']>=c['upper']-eps)
        result={k:v for k,v in result.items() if k!='inverse'}
        result.setdefault('hessian_edm',jnp.float64(0. if not n else jnp.nan))
        result.setdefault('edm_curvature_validated',(result['status']==0)&(n==0))
        result.setdefault('edm_refinements',jnp.int32(0))
        result.setdefault('hessian_evaluations',jnp.int32(0))
        result['edm_refinement_enabled']=jnp.array(stopping_criterion=='edm' and edm_validation=='exact_hessian')
        result['errors']=jnp.sqrt(jnp.diag(result['covariance']))
        result['curvature_errors']=result['errors']
        result['converged']=result['status']==0
        result['valid_covariance']=(result['covariance_status']==0)|(result['covariance_status']==7)
        values=result['values'][c['free_indices']]
        reported,status=minuit_reported_errors(values,result['errors'][c['free_indices']],
            c['physical_lower'],c['physical_upper'],xp=jnp)
        full_reported=jnp.zeros(total,dtype=jnp.float64).at[c['free_indices']].set(reported,unique_indices=True)
        full_status=jnp.ones(total,dtype=status.dtype).at[c['free_indices']].set(status,unique_indices=True)
        result['reported_errors']=jnp.where(result['valid_covariance'],full_reported,jnp.nan)
        result['reported_error_status']=jnp.where(result['valid_covariance'],full_status,3)
        return result
    return fit


def prepare_fit(model, *, initial=None, fixed=None, **options):
    """Validate neutral metadata, prepare numeric arguments, and return a JIT.

    Returns ``(compiled_function, coordinates, model_arrays)``. Dispatch with
    ``compiled_function(coordinates, model_arrays)``; results remain device arrays.
    Executable-cache policy and synchronization belong to the caller.
    """
    import jax
    from .model import make_model_function
    from .parameter_info import model_parameter_info,fit_coordinates
    c=fit_coordinates(model_parameter_info(model),initial=initial,fixed=fixed)
    coordinates={n:c[n] for n in COORDINATE_FIELDS}
    # Fit metadata is transported with the model but is not an NLL operand.
    arrays={n:a for n,a in model.arrays.items() if not n.startswith('parameter_info__')}
    function=jax.jit(make_fit_function(make_model_function(model),**options))
    return function,jax.device_put(coordinates),jax.device_put(arrays)
