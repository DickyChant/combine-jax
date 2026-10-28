"""Small, framework-neutral fit-state views for native profile comparisons."""
import numpy as np

from .bundle import TensorBundle
from .parameter_info import fit_coordinates, model_parameter_info


def configured_profile_model(model, fixed):
    """Freeze physical coordinates without reimporting or copying model tensors.

    The original model remains immutable. Its likelihood arrays, parameter
    order, dataset provenance and operation schedule are preserved; only fit
    starts, fixed-coordinate checks and the expected native fit state change.
    Native workspace preparation requires values inside the stored ranges.
    """
    info=model_parameter_info(model)
    policy=info.metadata.get('fit_policy')
    if policy is None:
        raise ValueError('Native profile comparison requires an explicit fit policy')
    fixed={name:float(value) for name,value in dict(fixed).items()}
    if not fixed:
        raise ValueError('A profile point needs at least one fixed parameter')
    coordinates=fit_coordinates(info,fixed=fixed)
    initial=coordinates['reference']
    if np.any(initial<info.arrays['stored_lower']) or np.any(initial>info.arrays['stored_upper']):
        raise ValueError('Native profile point exceeds stored workspace bounds')
    arrays=dict(model.arrays)
    for field in ('initial','lower','upper','stored_lower','stored_upper'):
        value=info.arrays[field].copy()
        for name,point in fixed.items():value[info.metadata['names'].index(name)]=point
        value.flags.writeable=False
        arrays['parameter_info__'+field]=value
    arrays['scalars__parameter_initial']=arrays['parameter_info__initial']
    frozen=arrays['parameter_info__lower']==arrays['parameter_info__upper']
    frozen.flags.writeable=False
    arrays['scalars__parameter_fixed']=frozen
    # Scalar programs put immutable coordinates into their initial slot buffer.
    indices=model.arrays['scalars__parameter_indices']
    slots=model.arrays['scalars__parameter_slots']
    keep=~frozen[indices]
    values=model.arrays['scalars__initial_values'].copy()
    values[slots[~keep]]=initial[indices[~keep]]
    for name,array in (('initial_values',values),('parameter_indices',indices[keep]),
                       ('parameter_slots',slots[keep])):
        array.flags.writeable=False
        arrays['scalars__'+name]=array
    state={**policy,'fixed_real_values':{**policy['fixed_real_values'],**fixed},
           'nuisance_bounds_removed':[name for name in policy['nuisance_bounds_removed'] if name not in fixed]}
    result=TensorBundle._from_frozen({**model.metadata,
        'parameter_info':{**info.metadata,'fit_policy':state},
        'profile_point':dict(fixed=fixed,other_parameters_start='imported nominal')},arrays)
    model_parameter_info(result)
    return result


def apply_native_profile_point(root, workspace, model):
    """Mirror Combine's explicit freeze operation at the ROOT boundary.

    Combine removes frozen nuisances from both model configurations as well as
    setting their constant flags. That changes the bounds-optimization list;
    merely setting a RooRealVar constant is not the CLI's actual fit state.
    """
    fixed=model.metadata['profile_point']['fixed']
    for name,value in fixed.items():
        variable=workspace.var(name)
        if not variable:
            raise ValueError(f'Profile parameter missing from workspace: {name}')
        variable.setVal(value);variable.setConstant(True)
    config=workspace.obj('ModelConfig')
    if not config:raise ValueError('Missing ModelConfig')
    original=config.GetNuisanceParameters()
    nuisance=root.RooArgSet(original) if original else root.RooArgSet()
    for parameter in fixed:nuisance.remove(workspace.var(parameter),True,True)
    config.SetNuisanceParameters(nuisance)
    background=workspace.obj('ModelConfig_bonly')
    if background:background.SetNuisanceParameters(nuisance)
