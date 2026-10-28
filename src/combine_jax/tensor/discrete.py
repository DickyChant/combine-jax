"""Neutral active-parameter maps and compact channels for discrete scans."""
import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle


def attach_discrete_profile(model,nodes,rows,scalars,*,constraint_names,compact_data=True):
    """Retain one small channel program and masks, never one model per choice.

    The initial implementation covers independent, unconstrained MultiPdf shape
    parameters. Common signal/yield/constraint parameters remain in the full
    fit. Shared discrete shape parameters need native masking-offset semantics
    beyond this factorized short scan and are rejected explicitly.
    """
    from .binned_channels import pack_binned_channels
    binned=model.metadata['components'].get('binned',{})
    categories=binned.get('discrete_categories',[])
    if not categories:return model
    names=model.metadata['parameter_names'];lookup={n:i for i,n in enumerate(names)}
    cache={}
    def dependencies(name):
        if name not in cache:
            cache[name]={name}|set().union(*(dependencies(n) for n in nodes[name][1]))
        return cache[name]
    k=len(categories);width=max(c['count'] for c in categories)
    masks=np.zeros((k,width,len(names)),dtype=bool)
    category_lookup={c['name']:i for i,c in enumerate(categories)}
    alternatives={}
    for name,(cls,_,attrs) in nodes.items():
        if cls!='RooMultiPdf' or not attrs.get('profiled'):continue
        i=category_lookup[attrs['category']]
        alternatives[name]=[]
        for index,child in enumerate(attrs['alternatives']):
            active=sorted(set(lookup)&dependencies(child))
            nested=[n for n in dependencies(child) if nodes[n][0]=='RooMultiPdf']
            if nested:raise UnsupportedModelError(f'{name}: nested discrete PDFs need explicit native profiling validation')
            masks[i,index,[lookup[n] for n in active]]=True
            alternatives[name].append(active)
    union=np.any(masks,axis=(0,1))
    if np.any(np.sum(np.any(masks,axis=1),axis=0)>1):
        raise UnsupportedModelError('Discrete shape parameters shared between categories need native masking-offset validation')
    union_names={n for n,i in lookup.items() if union[i]}
    constrained=union_names&set().union(*(dependencies(n) for n in constraint_names))
    if constrained:raise UnsupportedModelError(f'Constrained discrete shape parameters need native masking-offset validation: {sorted(constrained)}')
    selected=[];membership=[]
    for row in rows:
        deps=dependencies(row['pdf'])
        active=[c['name'] in deps for c in categories]
        if any(active) and not row['masked']:
            selected.append(row);membership.append(active)
        elif not row['masked'] and union_names&deps:
            raise UnsupportedModelError(f'{row["name"]}: discrete shape parameters also affect a non-discrete channel')
    subset=pack_binned_channels(nodes,selected,scalars,compact_data=compact_data)
    if subset.metadata['discrete_categories']!=categories:
        raise ValueError('Discrete subset category order differs from the full model')
    offsets=[np.zeros(k,dtype=np.int32)]
    for i,category in enumerate(categories):
        for step in range(1,category['count']):
            row=np.zeros(k,dtype=np.int32);row[i]=step;offsets.append(row)
    arrays={**model.arrays,**{'discrete_channels__'+n:a for n,a in subset.arrays.items()},
        'discrete_profile__alternative_parameters':masks,
        'discrete_profile__parameter_union':union,
        'discrete_profile__channel_categories':np.asarray(membership,dtype=bool),
        'discrete_profile__orthogonal_offsets':np.stack(offsets),
        'discrete_profile__category_sizes':np.asarray([c['count'] for c in categories],dtype=np.int32)}
    metadata={**model.metadata,'discrete_profile':dict(categories=categories,
        channels=subset.metadata,alternative_parameters=alternatives,
        scope='Native short combinations, independent unconstrained shape parameters; common parameters frozen during the discrete pass.')}
    # Preserve ownership of large model buffers; only the new maps need freezing.
    for value in arrays.values():value.flags.writeable=False
    return TensorBundle._from_frozen(metadata,arrays)


def active_parameters(indices,arrays):
    """Full-fit mask: common parameters plus the current alternative's union."""
    import jax.numpy as jnp
    table=arrays['discrete_profile__alternative_parameters']
    return ~arrays['discrete_profile__parameter_union']|jnp.any(table[jnp.arange(len(indices)),indices],axis=0)


def make_discrete_channels_function(model):
    """Return the small per-channel short-scan likelihood, without constraints."""
    from .jax_scalar import make_function
    from .binned_channels import make_binned_function
    scalar=TensorBundle._from_frozen(model.metadata['components']['scalars'],
        {n[len('scalars__'):]:v for n,v in model.arrays.items() if n.startswith('scalars__')})
    subset=TensorBundle._from_frozen(model.metadata['discrete_profile']['channels'],
        {n[len('discrete_channels__'):]:v for n,v in model.arrays.items() if n.startswith('discrete_channels__')})
    sf=make_function(scalar);bf=make_binned_function(subset,gradient='structural_sequential')
    def channels(theta,arrays):
        values=sf(theta,{n:arrays['scalars__'+n] for n in scalar.arrays})
        data={n:arrays['discrete_channels__'+n] for n in subset.arrays}
        data['discrete_indices']=arrays['binned__discrete_indices']
        return bf(values,data)
    return channels
