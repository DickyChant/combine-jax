"""Batched CMSHistFunc channels, statistical shifts and exact analytic bin profiles."""
from collections import defaultdict
import math

import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle
from .scalar import validate_program


def smooth_step(theta,region,xp=np):
    x=theta/region
    polynomial=x*(x*x*(3*x*x-10)+15)/8
    return xp.where(xp.abs(theta)>=region,xp.where(theta>0,1.,-1.),polynomial)


def morph_batch(mode,theta,buffers,xp=np):
    theta=xp.where(buffers['morph_mask'],theta,0.)
    alpha=smooth_step(theta,buffers['region'][:,None],xp)
    delta=.5*theta[:,:,None]*(buffers['diff']+alpha[:,:,None]*buffers['sum'])
    bins=buffers['base']+xp.sum(delta,axis=1)
    if mode=='log':
        bins=xp.where(buffers['source_mask'],xp.exp(bins),0.)
        bins=bins*(buffers['nominal_sum']/xp.sum(bins,axis=1))[:,None]
    elif mode!='linear':
        raise ValueError(f'Unknown CMSHistFunc mode {mode}')
    bins=xp.where(buffers['source_mask'],xp.maximum(bins,1e-9),0.)
    return xp.einsum('hbf,hf->hb',buffers['projection'],bins)+buffers['tail']


def profile_solution(mean,error,observed,global_observed,lower,upper,xp=np):
    safe_error=xp.where(error>0,error,1.)
    mean=xp.where(error>0,mean,0.)
    observed=xp.where(error>0,observed,0.)
    global_observed=xp.where(error>0,global_observed,0.)
    b=safe_error+mean/safe_error-global_observed
    c=mean-observed-(mean/safe_error)*global_observed
    first=-.5*(b+xp.copysign(1.,b)*xp.sqrt(b*b-4*c))
    second=c/xp.where(first!=0,first,1.)
    solution=xp.maximum(first,second)
    valid=(error>0)&(solution>=lower)&(solution<=upper)
    return xp.where(valid,solution,xp.nan)


def template_terms(histograms,scalar_values,arrays,xp=np):
    """All channels at once; no per-bin, per-process or per-profile Python loop."""
    process_mask=arrays['process_mask']
    values=xp.where(process_mask[:,:,None],histograms[arrays['histogram_indices']],0.)
    coefficients=xp.where(process_mask,scalar_values[arrays['coefficient_indices']],0.)
    total=xp.sum(coefficients[:,:,None]*values,axis=1)
    variance=xp.sum((coefficients[:,:,None]*arrays['errors'])**2,axis=1)
    error=xp.where(variance>0,xp.sqrt(xp.where(variance>0,variance,1.)),0.)
    pc,pb=arrays['profile_channels'],arrays['profile_bins']
    profiles=profile_solution(total[pc,pb]*arrays['profile_widths'],
        error[pc,pb]*arrays['profile_widths'],arrays['profile_observed'],arrays['profile_globals'],
        arrays['profile_minimum'],arrays['profile_maximum'],xp)
    combined=xp.concatenate((scalar_values,profiles))
    parameters=combined[arrays['bin_parameter_indices']]
    kinds=arrays['bin_types']
    poisson=xp.where(kinds==2,coefficients[:,:,None]*values*(parameters-1.),0.)
    gaussian=xp.where(kinds==3,coefficients[:,:,None]*arrays['errors']*parameters,0.)
    shift=xp.sum(poisson+gaussian,axis=1)
    shift=shift+xp.where(kinds[:,0,:]==1,error*parameters[:,0,:],0.)
    corrected=xp.maximum(total+shift,1e-9)
    corrected=xp.where(arrays['active_mask'],corrected,arrays['propagator_tail'])
    bins=xp.where(arrays['propagator_mode'][:,None],corrected,total)
    outer=xp.where(arrays['propagator_mode'],scalar_values[arrays['outer_indices']],1.)
    bins=bins*outer[:,None]
    expected=xp.sum(xp.where(arrays['active_mask'],bins*arrays['widths'],0.),axis=1)
    intensity=xp.take_along_axis(bins,arrays['data_bins'],axis=1)
    intensity=xp.where(arrays['data_valid'],intensity,0.)
    weights=arrays['data_weights']
    observed=xp.sum(weights,axis=1)
    density=xp.where(weights==0,1.,intensity/expected[:,None])
    nll=-xp.sum(weights*xp.log(density),axis=1)+expected-observed-observed*(
        xp.log(expected)-xp.log(xp.where(observed>0,observed,1.)))
    return nll,profiles,bins


def _histogram_record(name,attrs,lookup,dynamic,constant):
    region=float(attrs['smooth_region'])
    mode={0:'linear',1:'log'}.get(attrs['vertical_type'])
    active=int(attrs.get('source_active',attrs['active']))
    source=np.asarray(attrs['templates'],dtype=np.float64)
    if mode is None or not math.isfinite(region) or region<=0 or active<=0:
        raise UnsupportedModelError(f'{name}: invalid histogram morph configuration')
    if source.ndim!=2 or source.shape[0]!=1+2*len(attrs['morphs']) or active>source.shape[1]:
        raise UnsupportedModelError(f'{name}: invalid template storage/active range')
    if not np.all(np.isfinite(source)):
        raise UnsupportedModelError(f'{name}: nonfinite template contents')
    nominal=source[0,:active]
    if mode=='log':
        base=np.where(nominal>0,np.log(np.where(nominal>0,nominal,1.)),-999.)
    else:base=nominal.copy()
    diffs,sums,indices=[],[],[]
    for i,n in enumerate(attrs['morphs']):
        if n not in lookup:raise UnsupportedModelError(f'{name}: missing scalar morph binding {n}')
        lo,hi=source[2*i+1,:active],source[2*i+2,:active]
        if mode=='log':
            def log_ratio(value):
                valid=(value>0)&(nominal>0)
                return np.where(valid,np.log(np.where(valid,value,1.)/np.where(valid,nominal,1.)),0.)
            lo,hi=log_ratio(lo),log_ratio(hi)
        else:lo,hi=lo-nominal,hi-nominal
        diff,summ=hi-lo,hi+lo
        if n in dynamic:
            indices.append(lookup[n]);diffs.append(diff);sums.append(summ)
        else:
            theta=constant[n]
            base+=.5*theta*(diff+smooth_step(theta,region)*summ)
    edges=np.asarray(attrs['edges'],dtype=np.float64)
    cache=np.asarray(attrs['cache'],dtype=np.float64)
    if edges.shape!=(len(cache)+1,) or not np.all(np.diff(edges)>0):
        raise UnsupportedModelError(f'{name}: invalid output geometry')
    projection=np.zeros((len(cache),active))
    if 'rebin_indices' in attrs:
        mapping=np.asarray(attrs['rebin_indices'],dtype=np.int32)[:active]
        widths=np.diff(attrs['source_edges'])[:active]
        if np.any(mapping<0) or np.any(mapping>=attrs['active']):
            raise UnsupportedModelError(f'{name}: invalid rebinning map')
        projection[mapping,np.arange(active)]=widths/np.diff(edges)[mapping]
        tail=np.zeros_like(cache)
    else:
        projection[np.arange(active),np.arange(active)]=1.
        tail=cache.copy();tail[:active]=0.
    return dict(name=name,mode=mode,base=base,diff=np.array(diffs).reshape(len(indices),active),
        sum=np.array(sums).reshape(len(indices),active),indices=indices,region=region,
        nominal_sum=float(np.sum(nominal)),projection=projection,tail=tail,
        source_active=active,output_size=len(cache))


def pack_templates(nodes,channels,scalars,*,profiles=()):
    """Pack aligned CMSHistFunc sums and error propagators; profiles are declarations.

    Profile observed values/widths/bounds are checked again against actual channel
    rows and node metadata. This imports no JAX or ROOT and solves no fit.
    """
    validate_program(scalars)
    scalar_names=scalars.metadata['output_names'];lookup={n:i for i,n in enumerate(scalar_names)}
    dynamic_slots=set(scalars.arrays['parameter_slots'].tolist())
    for batch in scalars.metadata['schedule']:dynamic_slots.update(scalars.arrays[batch['outputs']].tolist())
    dynamic={n for n,s in zip(scalar_names,scalars.arrays['output_slots'],strict=True) if int(s) in dynamic_slots}
    constant={n:float(scalars.arrays['initial_values'][s]) for n,s in zip(scalar_names,scalars.arrays['output_slots'],strict=True) if n not in dynamic}
    profiles=[dict(p) if isinstance(p,dict) else vars(p) for p in profiles]
    profile_by_prop={p['propagator']:p for p in profiles}
    profile_names=sorted(n for p in profiles for n in p['parameters'])
    if len(set(profile_names))!=len(profile_names) or len(profile_by_prop)!=len(profiles) or set(profile_names)&set(scalar_names):
        raise UnsupportedModelError('Duplicate/shared or ambiguous analytic profile binding')
    combined={**lookup,**{n:len(scalar_names)+i for i,n in enumerate(profile_names)}}
    histograms={};records=[];used_profiles={}
    for channel in channels:
        label=channel['name'];cls,_,root=nodes[channel['pdf']]
        if channel.get('masked',False):raise UnsupportedModelError(f'{label}: masked channels must be folded before template grouping')
        if cls!='RooRealSumPdf' or not root['extended']:
            raise UnsupportedModelError(f'{label}: expected an extended RooRealSumPdf')
        functions=root['functions'];coefs=root['coefs'];propagator=None
        if len(functions)==1 and nodes[functions[0]][0]=='CMSHistErrorPropagator':
            propagator=functions[0];prop=nodes[propagator][2]
            functions,coefs=prop['functions'],prop['coefs']
            outer=root['coefs'][0]
            if outer not in lookup:raise UnsupportedModelError(f'{label}: missing outer coefficient {outer}')
            if propagator in profile_by_prop and (outer in dynamic or constant[outer]!=1.):
                raise UnsupportedModelError(f'{label}: profiled propagator requires a fixed unit outer coefficient')
        else:
            prop=dict(bin_types=(),bin_parameters=());outer=None
        if not functions or len(functions)!=len(coefs):raise UnsupportedModelError(f'{label}: invalid template components')
        if any(nodes[n][0]!='CMSHistFunc' for n in functions):
            raise UnsupportedModelError(f'{label}: unsupported real-sum function class')
        if any(n not in lookup for n in coefs):raise UnsupportedModelError(f'{label}: missing scalar rate binding')
        first=nodes[functions[0]][2];axis=first['observable_name']
        if tuple(channel['observables'])!=(axis,):raise UnsupportedModelError(f'{label}: template channel requires its direct observable')
        for n in functions:
            a=nodes[n][2]
            if a['active']!=first['active'] or a['observable_name']!=axis or not np.array_equal(a['edges'],first['edges']):
                raise UnsupportedModelError(f'{label}: template component geometry differs')
            if len(a['errors'])<a['active'] or not np.all(np.isfinite(a['errors'])):
                raise UnsupportedModelError(f'{label}: invalid histogram errors')
            if n not in histograms:histograms[n]=_histogram_record(n,a,lookup,dynamic,constant)
        bcount=len(first['cache']);active=int(first['active'])
        if not 0<active<=bcount:raise UnsupportedModelError(f'{label}: invalid active output bins')
        coords=np.asarray(channel['coordinates'][axis]);weights=np.asarray(channel['weights'],dtype=np.float64)
        if coords.ndim!=1 or coords.shape!=weights.shape or not np.all(np.isfinite(coords)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
            raise UnsupportedModelError(f'{label}: invalid weighted data')
        data_bins=np.searchsorted(first['edges'],coords,side='left')-1
        data_valid=(data_bins>=0)&(data_bins<bcount)
        types=prop['bin_types'];params=prop['bin_parameters']
        if len(types) not in (0,active) or len(params)!=len(types):raise UnsupportedModelError(f'{label}: invalid bin-statistics layout')
        kind=np.zeros((len(functions),bcount),dtype=np.int32)
        indices=np.zeros_like(kind)
        for b,(ts,ps) in enumerate(zip(types,params,strict=True)):
            if not ts or len(ts)!=len(ps) or any(t not in range(5) for t in ts):raise UnsupportedModelError(f'{label}: invalid bin types')
            if (ts[0] in (0,1) and len(ts)!=1) or (ts[0] not in (0,1) and len(ts)!=len(functions)):
                raise UnsupportedModelError(f'{label}: invalid per-process bin types')
            for j,(t,p) in enumerate(zip(ts,ps,strict=True)):
                kind[j,b]=t
                if t in (1,2,3):
                    if p not in combined:raise UnsupportedModelError(f'{label}: missing bin parameter {p}')
                    indices[j,b]=combined[p]
        if propagator in profile_by_prop:
            if not np.all(data_valid):raise UnsupportedModelError(f'{label}: profiled data outside histogram geometry')
            declaration=profile_by_prop[propagator]
            observed=np.zeros(bcount)
            for b,w in zip(data_bins,weights,strict=True):observed[b]=w  # Native setData assigns the last row.
            fields=['indices','parameters','global_observed','observed','widths','minimum','maximum']
            if len({len(declaration[k]) for k in fields})!=1:raise ValueError('Profile declaration lengths differ')
            for b,n,g,o,w,lo,hi in zip(*(declaration[k] for k in fields),strict=True):
                if b<0 or b>=active or tuple(types[b])!=(1,) or tuple(params[b])!=(n,) or nodes[n][0]!='RooRealVar':
                    raise UnsupportedModelError(f'{label}: invalid profile bin declaration {n}')
                if o!=observed[b] or w!=np.diff(first['edges'])[b] or lo!=nodes[n][2]['minimum'] or hi!=nodes[n][2]['maximum']:
                    raise UnsupportedModelError(f'{label}: profile declaration differs from data/geometry/bounds')
                global_node=nodes.get(n+'_In')
                if global_node is None or not global_node[2].get('constant',global_node[0]=='RooConstVar') or global_node[2]['value']!=g:
                    raise UnsupportedModelError(f'{label}: profile global observation differs from declaration')
                if n in used_profiles:raise UnsupportedModelError(f'{n}: shared analytic profile')
                used_profiles[n]=dict(channel=len(records),bin=b,global_observed=g,observed=o,width=w,minimum=lo,maximum=hi)
        records.append(dict(name=label,functions=functions,coefs=coefs,outer=outer,propagator=propagator,
            active=active,bcount=bcount,edges=np.array(first['edges']),tail=np.array(first['cache']),
            errors=[np.asarray(nodes[n][2]['errors']) for n in functions],types=kind,parameters=indices,
            data_bins=np.clip(data_bins,0,bcount-1),data_valid=data_valid,weights=weights))
    if set(used_profiles)!=set(profile_names):raise UnsupportedModelError('Profile declarations not covered exactly once by template channels')
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate template channel name')
    if not records:raise ValueError('No template channels to pack')
    hnames=list(histograms);hlookup={n:i for i,n in enumerate(hnames)}
    B=max(r['bcount'] for r in records);P=max(len(r['functions']) for r in records);R=max(len(r['weights']) for r in records);C=len(records)
    initial=np.zeros((len(hnames),B));groups=defaultdict(list);folded=[]
    for n,h in histograms.items():
        if h['indices']:groups[h['mode']].append(h)
        else:
            buffers=dict(base=h['base'][None,:],diff=h['diff'][None,:,:],sum=h['sum'][None,:,:],
                morph_mask=np.empty((1,0),dtype=bool),region=np.array([h['region']]),source_mask=np.ones((1,h['source_active']),dtype=bool),
                nominal_sum=np.array([h['nominal_sum']]),projection=h['projection'][None,:,:],tail=h['tail'][None,:])
            values=morph_batch(h['mode'],np.empty((1,0)),buffers)[0]
            if not np.all(np.isfinite(values)):raise UnsupportedModelError(f'{n}: nonfinite fixed histogram')
            initial[hlookup[n],:h['output_size']]=values;folded.append(n)
    arrays={'histogram_values':initial};schedule=[]
    for gi,(mode,group) in enumerate(sorted(groups.items())):
        D=len(group);F=max(h['source_active'] for h in group);M=max(len(h['indices']) for h in group)
        fields=dict(base=np.zeros((D,F)),diff=np.zeros((D,M,F)),sum=np.zeros((D,M,F)),
            morph_mask=np.zeros((D,M),dtype=bool),region=np.ones(D),source_mask=np.zeros((D,F),dtype=bool),
            nominal_sum=np.zeros(D),projection=np.zeros((D,B,F)),tail=np.zeros((D,B)))
        inputs=np.zeros((D,M),dtype=np.int32);outputs=np.array([hlookup[h['name']] for h in group],dtype=np.int32)
        for i,h in enumerate(group):
            f,m,b=h['source_active'],len(h['indices']),h['output_size']
            inputs[i,:m]=h['indices'];fields['morph_mask'][i,:m]=True;fields['source_mask'][i,:f]=True
            fields['base'][i,:f]=h['base'];fields['diff'][i,:m,:f]=h['diff'];fields['sum'][i,:m,:f]=h['sum']
            fields['region'][i]=h['region'];fields['nominal_sum'][i]=h['nominal_sum']
            fields['projection'][i,:b,:f]=h['projection'];fields['tail'][i,:b]=h['tail']
        arrays[f'morph_inputs_{gi}']=inputs;arrays[f'morph_outputs_{gi}']=outputs
        bindings={k:f'morph_{gi}_{k}' for k in fields}
        arrays.update({bindings[k]:v for k,v in fields.items()})
        schedule.append(dict(operation='cms_histfunc',mode=mode,inputs=f'morph_inputs_{gi}',outputs=f'morph_outputs_{gi}',buffers=bindings))
    arrays.update(histogram_indices=np.zeros((C,P),dtype=np.int32),coefficient_indices=np.zeros((C,P),dtype=np.int32),
        process_mask=np.zeros((C,P),dtype=bool),errors=np.zeros((C,P,B)),bin_types=np.zeros((C,P,B),dtype=np.int32),
        bin_parameter_indices=np.zeros((C,P,B),dtype=np.int32),active_mask=np.zeros((C,B),dtype=bool),
        widths=np.ones((C,B)),propagator_tail=np.zeros((C,B)),propagator_mode=np.zeros(C,dtype=bool),
        outer_indices=np.zeros(C,dtype=np.int32),data_bins=np.zeros((C,R),dtype=np.int32),
        data_valid=np.zeros((C,R),dtype=bool),data_weights=np.zeros((C,R)))
    for i,r in enumerate(records):
        p,b,rows=len(r['functions']),r['bcount'],len(r['weights'])
        arrays['histogram_indices'][i,:p]=[hlookup[n] for n in r['functions']]
        arrays['coefficient_indices'][i,:p]=[lookup[n] for n in r['coefs']];arrays['process_mask'][i,:p]=True
        for j,e in enumerate(r['errors']):arrays['errors'][i,j,:r['active']]=e[:r['active']]
        arrays['bin_types'][i,:p,:b]=r['types'];arrays['bin_parameter_indices'][i,:p,:b]=r['parameters']
        arrays['active_mask'][i,:r['active']]=True;arrays['widths'][i,:b]=np.diff(r['edges'])
        arrays['propagator_tail'][i,:b]=r['tail'];arrays['propagator_mode'][i]=r['propagator'] is not None
        if r['outer'] is not None:arrays['outer_indices'][i]=lookup[r['outer']]
        arrays['data_bins'][i,:rows]=r['data_bins'];arrays['data_valid'][i,:rows]=r['data_valid'];arrays['data_weights'][i,:rows]=r['weights']
    for field,source,dtype in [('channels','channel',np.int32),('bins','bin',np.int32),('widths','width',np.float64),
        ('observed','observed',np.float64),('globals','global_observed',np.float64),('minimum','minimum',np.float64),('maximum','maximum',np.float64)]:
        arrays['profile_'+field]=np.array([used_profiles[n][source] for n in profile_names],dtype=dtype)
    schedule.append(dict(operation='histogram_channels_and_profiles'))
    result=TensorBundle(dict(schema='combine-jax.tensor.v1',kind='template_channels',schedule=schedule,
        names=[r['name'] for r in records],histogram_names=hnames,folded_histograms=folded,
        scalar_names=scalar_names,profile_names=profile_names,
        scope='Aligned CMSHistFunc sums/propagators; native stored morph and rebin semantics; profiles solved on device with original bounds.'),arrays)
    validate_templates(result)
    return result


def validate_templates(bundle):
    m,a=bundle.metadata,bundle.arrays
    if m['kind']!='template_channels' or m['schedule'][-1]!={'operation':'histogram_channels_and_profiles'}:
        raise ValueError('Invalid template program')
    H,B=a['histogram_values'].shape;C,P=a['histogram_indices'].shape;R=a['data_bins'].shape[1]
    if H!=len(m['histogram_names']) or C!=len(m['names']) or not C or not P or not B:
        raise ValueError('Template layout mismatch')
    if len(set(m['names']))!=C or len(set(m['profile_names']))!=len(m['profile_names']):raise ValueError('Duplicate template/profile names')
    sizes=dict(histogram_indices=(C,P),coefficient_indices=(C,P),process_mask=(C,P),errors=(C,P,B),
        bin_types=(C,P,B),bin_parameter_indices=(C,P,B),active_mask=(C,B),widths=(C,B),propagator_tail=(C,B),
        propagator_mode=(C,),outer_indices=(C,),data_bins=(C,R),data_valid=(C,R),data_weights=(C,R))
    for n,shape in sizes.items():
        if a[n].shape!=shape:raise ValueError(f'Invalid template buffer {n}')
    for n in ('process_mask','active_mask','propagator_mode','data_valid'):
        if a[n].dtype!=bool:raise ValueError(f'Invalid mask {n}')
    for n in ('histogram_values','errors','widths','propagator_tail','data_weights'):
        if a[n].dtype!=np.float64 or not np.all(np.isfinite(a[n])):raise ValueError(f'Invalid numeric template buffer {n}')
    if np.any(a['widths']<=0) or np.any(a['data_weights']<0):raise ValueError('Invalid widths or weights')
    for n,limit in [('histogram_indices',H),('coefficient_indices',len(m['scalar_names'])),
        ('bin_parameter_indices',len(m['scalar_names'])+len(m['profile_names'])),('outer_indices',len(m['scalar_names'])),
        ('data_bins',B),('profile_channels',C),('profile_bins',B)]:
        if a[n].dtype!=np.int32 or np.any(a[n]<0) or np.any(a[n]>=limit):raise ValueError(f'Invalid gather {n}')
    if np.any((a['bin_types']<0)|(a['bin_types']>4)):raise ValueError('Unknown bin-statistics type')
    for n in ('channels','bins','widths','observed','globals','minimum','maximum'):
        if a['profile_'+n].shape!=(len(m['profile_names']),):raise ValueError('Profile metadata shape mismatch')
    for n in ('widths','observed','globals','minimum','maximum'):
        if a['profile_'+n].dtype!=np.float64 or np.any(np.isnan(a['profile_'+n])):raise ValueError('Invalid profile numeric buffer')
    if np.any(a['profile_widths']<=0) or np.any(a['profile_observed']<0) or np.any(a['profile_minimum']>a['profile_maximum']):
        raise ValueError('Invalid profile geometry/data/bounds')
    seen=[]
    for batch in m['schedule'][:-1]:
        if batch['operation']!='cms_histfunc' or batch['mode'] not in ('linear','log'):raise ValueError('Unknown morph operation')
        inputs=a[batch['inputs']];outputs=a[batch['outputs']]
        if inputs.dtype!=np.int32 or inputs.ndim!=2 or outputs.shape!=(len(inputs),) or outputs.dtype!=np.int32:
            raise ValueError('Invalid morph gather layout')
        if np.any(inputs<0) or np.any(inputs>=len(m['scalar_names'])) or np.any(outputs<0) or np.any(outputs>=H):raise ValueError('Morph gather out of range')
        buffers={k:a[n] for k,n in batch['buffers'].items()}
        D,M=inputs.shape;F=buffers['base'].shape[1]
        shapes=dict(base=(D,F),diff=(D,M,F),sum=(D,M,F),morph_mask=(D,M),region=(D,),source_mask=(D,F),
            nominal_sum=(D,),projection=(D,B,F),tail=(D,B))
        if set(buffers)!=set(shapes):raise ValueError('Invalid morph buffer names')
        for n,shape in shapes.items():
            dtype=bool if n.endswith('_mask') else np.float64
            if buffers[n].shape!=shape or buffers[n].dtype!=dtype or not np.all(np.isfinite(buffers[n])):
                raise ValueError(f'Invalid morph buffer {n}')
        if np.any(buffers['region']<=0):raise ValueError('Invalid morph smoothing region')
        seen.extend(outputs.tolist())
    if len(set(seen))!=len(seen):raise ValueError('Repeated dynamic histogram')
    folded=set(m['folded_histograms']);dynamic={m['histogram_names'][i] for i in seen}
    if folded&dynamic or folded|dynamic!=set(m['histogram_names']):raise ValueError('Histogram coverage mismatch')


def evaluate_templates(bundle,scalar_values):
    validate_templates(bundle)
    if np.shape(scalar_values)!=(len(bundle.metadata['scalar_names']),):raise ValueError('Scalar vector shape mismatch')
    histograms=bundle.arrays['histogram_values'].copy()
    for batch in bundle.metadata['schedule'][:-1]:
        buffers={k:bundle.arrays[n] for k,n in batch['buffers'].items()}
        histograms[bundle.arrays[batch['outputs']]]=morph_batch(batch['mode'],np.asarray(scalar_values)[bundle.arrays[batch['inputs']]],buffers)
    return template_terms(histograms,np.asarray(scalar_values),bundle.arrays)


def make_template_function(bundle):
    import jax.numpy as jnp
    validate_templates(bundle)
    schedule=tuple(dict(b) for b in bundle.metadata['schedule'][:-1]);size=len(bundle.metadata['scalar_names'])
    def evaluate(scalar_values,arrays):
        if scalar_values.shape!=(size,):raise ValueError('Scalar vector shape mismatch')
        histograms=arrays['histogram_values']
        for batch in schedule:
            buffers={k:arrays[n] for k,n in batch['buffers'].items()}
            values=morph_batch(batch['mode'],scalar_values[arrays[batch['inputs']]],buffers,jnp)
            histograms=histograms.at[arrays[batch['outputs']]].set(values)
        return template_terms(histograms,scalar_values,arrays,jnp)
    return evaluate
