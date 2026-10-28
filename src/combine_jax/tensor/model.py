"""Framework-neutral composition and execution of complete supported models."""
import numpy as np

from .bundle import TensorBundle
from .constraints import pack_constraints
from .counting import pack_counting_channels,make_counting_nll
from .scalar import SCALAR_CLASSES,dependency_flags,import_scalars


def compose_counting(scalars,constraints,channels,*,offset=0.,provenance=None):
    # This validates every component and refuses an omitted analytic-profile solver.
    from .constraints import validate_constraints
    from .counting import validate_counting
    from .scalar import validate_program
    validate_program(scalars);validate_constraints(constraints);validate_counting(channels)
    if constraints.metadata['profile_names']:
        raise ValueError('Counting model cannot omit profile-value computation')
    for component in (constraints,channels):
        if component.metadata['scalar_names'] != scalars.metadata['output_names']:
            raise ValueError('Component scalar bindings differ')
    components=dict(scalars=scalars,constraints=constraints,channels=channels)
    arrays={'offset':np.array(offset,dtype=np.float64,copy=True)}
    for name,component in components.items():
        arrays.update({name+'__'+n:a for n,a in component.arrays.items()})
    arrays['offset'].flags.writeable=False
    return TensorBundle._from_frozen(dict(schema='combine-jax.tensor.v1',kind='counting_likelihood',
        schedule=[dict(component=n,kind=c.metadata['kind'],schedule=c.metadata['schedule']) for n,c in components.items()],
        components={n:c.metadata for n,c in components.items()},
        parameter_names=scalars.metadata['parameter_names'],channel_names=channels.metadata['names'],
        constraint_names=constraints.metadata['names'],provenance=provenance or {},
        scope='Complete NLL for all channels/constraints supplied to this counting-model import.'),arrays)


def unpack_counting(model):
    if model.metadata['kind'] != 'counting_likelihood':
        raise ValueError('Expected counting likelihood')
    components={}
    for name in ('scalars','constraints','channels'):
        prefix=name+'__'
        arrays={n[len(prefix):]:a for n,a in model.arrays.items() if n.startswith(prefix)}
        components[name]=TensorBundle._from_frozen(model.metadata['components'][name],arrays)
    expected=[dict(component=n,kind=c.metadata['kind'],schedule=c.metadata['schedule']) for n,c in components.items()]
    if model.metadata['schedule'] != expected:
        raise ValueError('Composite and component schedules differ')
    return components


def compose_templates(scalars,constraints,channels,templates,*,offset=0.,provenance=None):
    """Compose all supplied terms; no analytic profile is supplied by the host."""
    from .constraints import validate_constraints
    from .counting import validate_counting
    from .scalar import validate_program
    from .templates import validate_templates
    validate_program(scalars);validate_constraints(constraints);validate_counting(channels);validate_templates(templates)
    if constraints.metadata['profile_names'] != templates.metadata['profile_names']:
        raise ValueError('Constraint and template profile bindings differ')
    for component in (constraints,channels,templates):
        if component.metadata['scalar_names'] != scalars.metadata['output_names']:
            raise ValueError('Component scalar bindings differ')
    names=channels.metadata['names']+templates.metadata['names']
    if len(set(names))!=len(names):raise ValueError('Duplicate model channel name')
    components=dict(scalars=scalars,constraints=constraints,channels=channels,templates=templates)
    arrays={'offset':np.array(offset,dtype=np.float64,copy=True)}
    for name,component in components.items():
        arrays.update({name+'__'+n:a for n,a in component.arrays.items()})
    arrays['offset'].flags.writeable=False
    return TensorBundle._from_frozen(dict(schema='combine-jax.tensor.v1',kind='template_likelihood',
        schedule=[dict(component=n,kind=c.metadata['kind'],schedule=c.metadata['schedule']) for n,c in components.items()],
        components={n:c.metadata for n,c in components.items()},parameter_names=scalars.metadata['parameter_names'],
        channel_names=names,constraint_names=constraints.metadata['names'],provenance=provenance or {},
        scope='Complete NLL for all supplied counting/template channels and constraints; profiles solved in the objective.'),arrays)


def unpack_model(model):
    if model.metadata['kind']=='counting_likelihood':return unpack_counting(model)
    kind=model.metadata['kind']
    if kind not in ('template_likelihood','binned_likelihood'):raise ValueError('Unknown likelihood kind')
    names=['scalars','constraints','channels']
    if 'templates' in model.metadata['components']:names.append('templates')
    if kind=='binned_likelihood':names.append('binned')
    components={n:TensorBundle._from_frozen(model.metadata['components'][n],
        {k[len(n)+2:]:a for k,a in model.arrays.items() if k.startswith(n+'__')})
        for n in names}
    expected=[dict(component=n,kind=c.metadata['kind'],schedule=c.metadata['schedule']) for n,c in components.items()]
    if model.metadata['schedule']!=expected:raise ValueError('Composite and component schedules differ')
    # Validate component layouts and cross-component bindings after transport.
    (compose_binned if kind=='binned_likelihood' else compose_templates)(**components,offset=model.arrays['offset'])
    return components


def compose_binned(scalars,constraints,channels,binned,*,templates=None,offset=0.,provenance=None):
    """Compose binned densities with the existing counting/template components."""
    from .binned_channels import validate_binned_channels
    validate_binned_channels(binned)
    if templates is None:base=compose_counting(scalars,constraints,channels,offset=offset,provenance=provenance)
    else:base=compose_templates(scalars,constraints,channels,templates,offset=offset,provenance=provenance)
    if binned.metadata['scalar_names']!=scalars.metadata['output_names']:raise ValueError('Component scalar bindings differ')
    names=base.metadata['channel_names']+binned.metadata['names']
    if len(names)!=len(set(names)):raise ValueError('Duplicate model channel name')
    metadata={**base.metadata,'kind':'binned_likelihood','channel_names':names,
        'components':{**base.metadata['components'],'binned':binned.metadata},
        'schedule':base.metadata['schedule']+[dict(component='binned',kind=binned.metadata['kind'],schedule=binned.metadata['schedule'])],
        'scope':'Complete NLL for all supplied counting, template and binned channels; profile and constraint coupling stays in the objective.'}
    arrays={**base.arrays,**{'binned__'+n:a for n,a in binned.arrays.items()}}
    return TensorBundle._from_frozen(metadata,arrays)


def make_model_function(model, *, binned_gradient='ad', row_chunk_size=1024):
    """Build an objective; structural modes differentiate parameters only.

    Model/data arrays are immutable runtime inputs. The default AD path retains
    general buffer differentiation; structural pullbacks provide derivatives
    with respect to the parameter vector, including its higher derivatives.
    """
    if binned_gradient not in ('ad', 'structural', 'structural_sequential'):
        raise ValueError('Unknown binned gradient policy')
    components=unpack_model(model)
    if 'templates' not in components and 'binned' not in components:
        function=make_counting_nll(**components)
    else:
        import jax.numpy as jnp
        from .jax_scalar import make_function
        from .constraints import make_constraint_function
        from .counting import make_counting_function
        from .templates import make_template_function
        from .binned_channels import make_binned_function
        sf=make_function(components['scalars']);cf=make_constraint_function(components['constraints'])
        chf=make_counting_function(components['channels'])
        tf=make_template_function(components['templates']) if 'templates' in components else None
        bf=make_binned_function(components['binned'],gradient=binned_gradient,
                                row_chunk_size=row_chunk_size) if 'binned' in components else None
        def function(theta,buffers):
            values=sf(theta,buffers['scalars'])
            profiles=jnp.empty((0,),dtype=jnp.float64)
            result=jnp.sum(chf(values,buffers['channels']))+buffers['offset']
            if tf is not None:
                template_nll,profiles,_=tf(values,buffers['templates']);result=result+jnp.sum(template_nll)
            if bf is not None:result=result+jnp.sum(bf(values,buffers['binned']))
            return result+jnp.sum(cf(values,profiles,buffers['constraints']))
    names={key:tuple(component.arrays) for key,component in components.items()}
    def nll(theta,arrays):
        buffers={key:{n:arrays[key+'__'+n] for n in fields} for key,fields in names.items()}
        buffers['offset']=arrays['offset']
        return function(theta,buffers)
    return nll


def evaluate_model_numpy(model,theta):
    from .scalar import evaluate_numpy
    from .constraints import evaluate_constraints
    from .counting import evaluate_counting
    components=unpack_model(model)
    values=evaluate_numpy(components['scalars'],theta)
    template_sum=0.;profiles=()
    if 'templates' in components:
        from .templates import evaluate_templates
        template_nll,profiles,_=evaluate_templates(components['templates'],values)
        template_sum=np.sum(template_nll)
    if 'binned' in components:
        from .binned_channels import evaluate_binned_channels
        template_sum+=np.sum(evaluate_binned_channels(components['binned'],values))
    constraints=evaluate_constraints(components['constraints'],values,profiles)
    channels=evaluate_counting(components['channels'],values)
    return np.sum(constraints)+np.sum(channels)+template_sum+model.arrays['offset']


def from_workspace(workspace,datacard=None,*,data='data_obs',analytic_barlow_beeston=True,compact_shape_data=True,fit_policy='workspace',template_convention='v10_density',discrete_policy='fixed'):
    """Complete supported workspace+card import; other shapes fail by name.

    Existing extraction is reused once at the boundary, including exact native
    parameter metadata, and is never invoked at numerical runtime.
    ``fit_policy='combine_v11_default'`` applies native effective nuisance ranges
    before extraction and preserves the workspace's mass. The legacy default
    retains the supplied workspace ranges for existing oracle comparisons.
    """
    if discrete_policy not in ('fixed','profile'):
        raise ValueError('Unknown discrete import policy')
    if fit_policy!='workspace':
        from .fit_policy import POLICY,combine_v11_fit_context,attach_fit_policy
        if fit_policy!=POLICY:raise ValueError(f'Unsupported fit policy: {fit_policy}')
        with combine_v11_fit_context(workspace) as (state,ranges):
            model=from_workspace(workspace,datacard,data=data,analytic_barlow_beeston=analytic_barlow_beeston,
                compact_shape_data=compact_shape_data,template_convention=template_convention,discrete_policy=discrete_policy)
        return attach_fit_policy(model,state,ranges)
    from optparse import OptionParser
    from pathlib import Path
    import hashlib
    from importlib.metadata import version
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard,addDatacardParserOptions
    from ..graph import walk
    from ..likelihood import from_workspace as oracle_export
    from ..root_io import load_root
    parser=OptionParser();addDatacardParserOptions(parser)
    options,_=parser.parse_args([])
    datacard=Path(datacard) if datacard is not None else None
    card=None
    if datacard is not None:
        with datacard.open() as stream:card=parseCard(stream,options)
    from ..translators import default_registry
    exported=oracle_export(workspace,data=data,analytic_barlow_beeston=analytic_barlow_beeston,
        registry=default_registry(template_convention=template_convention,profile_multipdf=discrete_policy=='profile'))
    if card is not None and set(card.bins) != {c.name for c in exported.channels}:
        raise ValueError('Datacard/workspace channel sets differ')
    nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in exported.nodes.items()}
    observables={n for c in exported.channels for n in c.observables}
    profiles=sorted(n for p in exported.profiles for n in p.parameters)
    flags=dependency_flags(nodes,exported.parameters,observables=observables,profiles=profiles)
    roots=sorted(n for n in nodes if nodes[n][0] in SCALAR_CLASSES and not any(flags[n][1:]))
    raw=walk(workspace.obj('ModelConfig').GetPdf())
    scalars=import_scalars(nodes,exported.parameters,roots,fixed_value=lambda n:float(raw[n].getVal()),
        observables=observables,profiles=profiles)
    root=load_root()
    def integral(name,axes):
        return float(raw[name].createIntegral(root.RooArgSet(*(raw[n] for n in axes))).getVal())
    constraints=pack_constraints(nodes,exported.constraints,exported.constraint_axes,scalars,
        profile_names=profiles,fixed_normalization=integral)
    rows=[dict(name=c.name,pdf=c.pdf,observables=c.observables,coordinates=dict(c.data.coordinates),
        weights=c.data.weights,masked=c.masked) for c in exported.channels]
    template_rows=[r for r in rows if not r['masked'] and nodes[r['pdf']][0]=='RooRealSumPdf']
    template_names={r['name'] for r in template_rows}
    def uniform(row):
        if row['masked']:return True
        cls,_,a=nodes[row['pdf']]
        return cls=='RooAddPdf' and all(nodes[n][0]=='RooUniform' for n in a['pdfs'])
    counting_rows=[r for r in rows if uniform(r)]
    binned_rows=[r for r in rows if r['name'] not in template_names and not uniform(r)]
    channels=pack_counting_channels(nodes,counting_rows,scalars)
    provenance=dict(card=str(datacard) if datacard is not None else None,
            card_sha256=hashlib.sha256(datacard.read_bytes()).hexdigest() if datacard is not None else None,
            combine_version=version('higgs-combine-tool'),root_version=str(root.gROOT.GetVersion()),
            workspace_name=str(workspace.GetName()),all_card_channels_imported=True if card is not None else None,
            channel_source='workspace_modelconfig' if card is None else 'workspace_and_datacard',
            template_convention=template_convention,
            discrete_choices=exported.discrete_choices,channel_masks=exported.mask_configuration)
    from ..dataset_identity import binding_identity
    provenance['dataset']=dict(name=data,identity=binding_identity({c.name:c.data for c in exported.channels}))
    from .templates import pack_templates
    templates=pack_templates(nodes,template_rows,scalars,profiles=exported.profiles) if template_rows else None
    if binned_rows:
        from .binned_channels import pack_binned_channels
        binned=pack_binned_channels(nodes,binned_rows,scalars,compact_data=compact_shape_data)
        model=compose_binned(scalars,constraints,channels,binned,templates=templates,offset=exported.offset,provenance=provenance)
    elif templates is not None:
        model=compose_templates(scalars,constraints,channels,templates,offset=exported.offset,provenance=provenance)
    else:
        model=compose_counting(scalars,constraints,channels,offset=exported.offset,provenance=provenance)
    if discrete_policy=='profile':
        from .discrete import attach_discrete_profile
        model=attach_discrete_profile(model,nodes,rows,scalars,
            constraint_names=exported.constraints,compact_data=compact_shape_data)
    from .parameter_info import attach_parameter_info,pack_parameter_info
    return attach_parameter_info(model,pack_parameter_info(exported.parameters))
