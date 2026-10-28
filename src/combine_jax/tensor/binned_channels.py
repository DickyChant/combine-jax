"""Batched binned densities, mixtures and conditional products.

The importer resolves normalization requests and folds fixed quantities. Runtime
programs batch by operator, dependency depth and data-row count, never RooFit node.
"""
from collections import defaultdict
import math

import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle
from .analytic_shapes import (ANALYTIC_CLASSES,ANALYTIC_OPERATIONS,GRID_ANALYTIC_OPERATIONS,pack_analytic_query,
    analytic_operation,validate_analytic_fields)
from .grid_expressions import GridBuilder,GRID_OPERATIONS,grid_operation,validate_grid_fields
from .shape_data import prepare_shape_data,validate_shape_data,padded_size
from .fast_histograms import (FAST_CLASSES,scalar_bindings,pack_fast_histograms,pack_fast_queries,
    make_fast_function,make_query_function,evaluate_fast_histograms,evaluate_fast_queries,
    validate_fast_histograms,validate_fast_queries)


BINNED_CLASSES=FAST_CLASSES|ANALYTIC_CLASSES|frozenset(('RooAddPdf','RooProdPdf','VerticalInterpPdf',
    'RooParametricHist2D','RooParametricStepFunction','RooHistPdf','RooUniform','RooMultiPdf'))


def density_children(node):
    cls,inputs,a=node
    if cls=='RooAddPdf':return a['pdfs']
    if cls=='RooProdPdf':return inputs
    if cls=='VerticalInterpPdf':return a['functions']
    if cls=='RooMultiPdf':return (a['selected'],)
    return ()


def supports_binned(nodes,name):
    cls,_,_=nodes[name]
    return cls in BINNED_CLASSES and all(supports_binned(nodes,n) for n in density_children(nodes[name]))


def product_normalizations(name,node,axes,depends):
    """Native factor ownership and acyclic conditional normalization semantics."""
    _,inputs,a=node;owners={};dependencies=[];normalizations=[]
    for i,(child,(kind,specified)) in enumerate(zip(inputs,a['normalisations'],strict=True)):
        if kind not in ('all','nset','cset'):raise UnsupportedModelError(f'{name}: unknown product normalization kind {kind}')
        dependent={v for v in axes if depends(child,v)}
        selected={v for v in specified if depends(child,v)}
        if kind=='cset':selected=dependent-selected
        normalized=dependent&selected if selected else dependent
        for v in normalized:
            if v in owners:raise UnsupportedModelError(f'{name}: overlapping product normalization over {v}')
            owners[v]=i
        dependencies.append(dependent);normalizations.append(tuple(sorted(normalized)))
    incoming=[{owners[v] for v in dep-set(norm) if v in owners} if norm else set()
        for dep,norm in zip(dependencies,normalizations,strict=True)]
    pending=set(range(len(inputs)))
    while pending:
        ready={i for i in pending if not incoming[i]&pending}
        if not ready:raise UnsupportedModelError(f'{name}: cyclic conditional product requires a joint integral')
        pending-=ready
    return normalizations


def operation(op,values,scalars,b,xp=np):
    """Rows have axes (batch, children or coefficients, data)."""
    if op in ANALYTIC_OPERATIONS:return analytic_operation(op,scalars,b,xp)
    if op in GRID_ANALYTIC_OPERATIONS:
        parameters=values[b['children']]
        if 'child_mask' in b:parameters=xp.where(b['child_mask'][:,:,None],parameters,0.)
        return analytic_operation(op,scalars,b,xp,parameters=parameters)
    if op in GRID_OPERATIONS:return grid_operation(op,values,b,xp)
    if op=='divide':return values[b['children'][:,0,:]]/values[b['children'][:,1,:]]
    if op=='product':
        return xp.prod(xp.where(b['child_mask'][:,:,None],values[b['children']],1.),axis=1)*b['factor'][:,None]
    coefs=xp.where(b['coef_mask'],scalars[b['coefs']],0.)
    if op=='coef_sum':return xp.sum(coefs,axis=1)[:,None]+b['broadcast']
    if op=='parametric_bins':
        selected=xp.take_along_axis(coefs,b['data_bins'],axis=1)/b['volume']
        return xp.where(b['data_valid'],selected,0.)
    if op=='step_bins':
        last=(1.-xp.sum(coefs*b['widths'],axis=1))/b['last_width']
        last=xp.where(last>0,last,1e-6)
        selected=xp.take_along_axis(coefs,b['data_bins'],axis=1)
        return xp.where(b['data_valid'],xp.where(b['last_bin'],last[:,None],selected),0.)
    if op in ('mixture_yields','mixture_fractions'):
        if op=='mixture_yields':weights=coefs/xp.sum(coefs,axis=1)[:,None]
        else:weights=coefs+b['last_component']*(1.-xp.sum(coefs,axis=1))[:,None]
        return xp.sum(xp.where(b['child_mask'][:,:,None],values[b['children']],0.)*weights[:,:,None],axis=1)
    if op.startswith('vertical_'):
        table=values[b['children']];central=table[:,0,:]
        high,low=table[:,1::2,:],table[:,2::2,:]
        theta=coefs[:,:,None];region=b['region'][:,None,None]
        safe=xp.where(region>0,region,1.)
        linear=theta*xp.where(theta>0,high-central[:,None,:],central[:,None,:]-low)
        if op=='vertical_0':
            up=theta*(region+theta)/(2*safe);down=-theta*(region-theta)/(2*safe)
            middle=up*high+down*low-theta**2/safe*central[:,None,:]
        elif op=='vertical_1':
            up=(region+theta)**2/(4*safe);down=(region-theta)**2/(4*safe)
            middle=up*high+down*low-(up+down)*central[:,None,:]
        elif op=='vertical_polynomial':
            z=theta/safe;alpha=z*(z*z*(3*z*z-10)+15)/8
            middle=.5*theta*((high-low)+alpha*(high+low-2*central[:,None,:]))
        else:raise ValueError(f'Unknown interpolation operation {op}')
        shift=xp.where((region==0)|(xp.abs(theta)>=region),linear,middle)
        result=central+xp.sum(xp.where(b['coef_mask'][:,:,None],shift,0.),axis=1)
        result=xp.where(result>0,result,b['pdf_floor'][:,None])
        return xp.where(result>0,result,b['second_floor'][:,None])
    raise ValueError(f'Unknown binned density operation {op}')


def nll_batch(values,scalars,b,xp=np):
    expected=xp.sum(xp.where(b['rate_mask'],scalars[b['rates']],0.),axis=1)
    observed=xp.sum(b['weights'],axis=1)
    density=xp.where(b['weights']==0,1.,values[b['densities']])
    return (-xp.sum(b['weights']*xp.log(density),axis=1)+expected-observed
        -observed*(xp.log(expected)-xp.log(xp.where(observed>0,observed,1.)))+b['correction'])


def _stack_records(records):
    """Pad child/coefficient axes; each operator's masks supply neutral entries."""
    fields={}
    for key in records[0]:
        arrays=[np.asarray(r[key]) for r in records]
        rank=arrays[0].ndim
        if any(a.ndim!=rank or a.dtype!=arrays[0].dtype for a in arrays):raise ValueError(f'Inconsistent operator field {key}')
        shape=tuple(max(a.shape[i] for a in arrays) for i in range(rank))
        result=np.zeros((len(arrays),*shape),dtype=arrays[0].dtype)
        for i,a in enumerate(arrays):result[(i,*(slice(0,n) for n in a.shape))]=a
        fields[key]=result
    return fields


def pack_binned_channels(nodes,channels,scalars,*,compact_data=True):
    channels,data_layouts,data_arrays=prepare_shape_data(channels,compact=compact_data)
    lookup,dynamic,constant=scalar_bindings(scalars)
    scalar_initial=np.array([constant.get(n,0.) for n in scalars.metadata['output_names']])
    fast_names=set();active_pdfs=set()
    def visit(name):
        if name in active_pdfs:return
        active_pdfs.add(name)
        cls,_,_=nodes[name]
        if cls not in BINNED_CLASSES:raise UnsupportedModelError(f'{name}: unsupported binned density class {cls}')
        if cls in FAST_CLASSES:fast_names.add(name)
        for child in density_children(nodes[name]):visit(child)
    for ch in channels:
        if not ch.get('masked',False):visit(ch['pdf'])
    histograms=pack_fast_histograms(nodes,sorted(fast_names),scalars)
    fixed_histograms=set(histograms.metadata['folded_histograms'])
    initial_chunks=[];initial_size=0;records=[];fast_requests=[];fast_bindings=[];cache={};folded=[];dependencies={};grids={};grid_queries=[]
    def depends(name,axis):
        key=name,axis
        if key not in dependencies:dependencies[key]=(name==axis or any(depends(n,axis) for n in nodes[name][1]))
        return dependencies[key]
    def scalar_fields(names,minimum=1):
        if any(n not in lookup for n in names):raise UnsupportedModelError(f'Observable/profile-dependent or missing scalar coefficients: {set(names)-lookup.keys()}')
        indices=np.zeros(max(len(names),minimum),dtype=np.int32);mask=np.zeros(len(indices),dtype=bool)
        indices[:len(names)]=[lookup[n] for n in names];mask[:len(names)]=True
        return dict(coefs=indices,coef_mask=mask)
    def allocate(rows):
        nonlocal initial_size
        start=initial_size;initial_size+=rows;initial_chunks.append(np.zeros(rows))
        return np.arange(start,initial_size,dtype=np.int32)
    def emit(op,rows,fields,children=(),coef_names=(),label=''):
        output=allocate(rows);depth=1+max((c[2] for c in children),default=-1)
        is_dynamic=any(c[1] for c in children) or any(n in dynamic for n in coef_names)
        if children:fields['children']=np.stack([c[0] for c in children])
        records.append(dict(op=op,rows=rows,fields=fields,outputs=output,dynamic=is_dynamic,depth=depth,label=label))
        return output,is_dynamic,depth
    def fixed(values,label):
        values=np.asarray(values,dtype=np.float64).reshape(-1);output=allocate(len(values))
        initial_chunks[-1][:]=values
        folded.append(label);return output,False,-1
    def analytic_grid(channel_index,ch,name,mode,axes,label):
        # Compress actual data coordinates, not an approximate integration grid.
        data_axes=tuple(sorted(v for v in ch['observables'] if depends(name,v) and (mode!='integral' or v not in axes)))
        key=channel_index,data_axes
        if key not in grids:
            R=len(ch['weights'])
            if data_axes:
                points,inverse=np.unique(np.stack([ch['coordinates'][v] for v in data_axes],axis=1),axis=0,return_inverse=True)
                distinct=len(points);count=padded_size(distinct) if compact_data else distinct
                if count>distinct:points=np.concatenate((points,np.repeat(points[:1],count-distinct,axis=0)))
                coords=dict(zip(data_axes,points.T,strict=True))
            else:coords={};count=distinct=min(R,1);inverse=np.zeros(R,dtype=np.int32)
            builder=GridBuilder(nodes,coords,count,lookup,dynamic,constant,emit,fixed,f'{ch["name"]}:grid:{data_axes}')
            grids[key]=(builder,coords,count,distinct,inverse)
        builder,coords,count,distinct,inverse=grids[key]
        coordinates={**coords,**{v:np.zeros(count) for v in axes if v not in coords}}
        op,fields,names=pack_analytic_query(name,nodes[name],mode,axes,coordinates,nodes,lookup,depends,count,allow_grid=True)
        op=op.replace('analytic_','gridanalytic_',1);fields={k:v for k,v in fields.items() if k not in ('coefs','coef_mask')}
        parameters=[builder.node(n) for n in names]
        if all(v.fixed is not None for v in parameters):
            params=np.stack([v.fixed for v in parameters])[None,...]
            value=analytic_operation(op,None,{k:v[None,...] for k,v in fields.items()},parameters=params)[0]
            if not np.all(np.isfinite(value)):raise UnsupportedModelError(f'{label}: invalid fixed analytic grid')
            result=fixed(value,label)
        else:
            fields['child_mask']=np.ones(len(parameters),dtype=bool)
            result=emit(op,count,fields,[builder.reference(v) for v in parameters],label=label)
        grid_queries.append(dict(name=label,coordinate_axes=data_axes,grid_rows=count,distinct_grid_rows=distinct,data_rows=len(inverse),folded=not result[1]))
        return result[0][inverse],result[1],result[2]
    def request(channel_index,ch,name,mode,axes):
        axes=tuple(sorted(axes));key=channel_index,name,mode,axes
        if key in cache:return cache[key]
        cls,inputs,a=nodes[name];R=len(ch['weights']);coords=ch['coordinates'];label=f'{ch["name"]}:{name}:{mode}:{axes}'
        if mode=='integral' and not axes:mode='value'
        if cls in FAST_CLASSES:
            output=allocate(R);query_name=f'q{len(fast_requests)}'
            fast_requests.append(dict(name=query_name,pdf=name,mode=mode,axes=axes,coordinates=coords))
            fast_bindings.append((output,query_name));result=output,name not in fixed_histograms,0
        elif cls=='RooMultiPdf':result=request(channel_index,ch,a['selected'],mode,axes)
        elif cls in ANALYTIC_CLASSES:
            if mode=='density':
                children=[request(channel_index,ch,name,m,axes) for m in ('value','integral')]
                result=emit('divide',R,{},children,label=label)
            else:
                op,fields,coefs=pack_analytic_query(name,nodes[name],mode,axes,coords,nodes,lookup,depends,R,allow_grid=True)
                if compact_data or any(n and n not in lookup for n in coefs):result=analytic_grid(channel_index,ch,name,mode,axes,label)
                else:result=emit(op,R,fields,coef_names=coefs,label=label)
        elif cls=='RooAddPdf':
            if mode!='density':raise UnsupportedModelError(f'{label}: RooAddPdf requires a normalization set')
            children=[request(channel_index,ch,n,'density',axes) for n in a['pdfs']]
            coefs=a['coefs'];fields=scalar_fields(coefs,minimum=len(children))
            fields['child_mask']=np.ones(len(children),dtype=bool)
            if not a['extended']:
                last=np.zeros(len(children));last[-1]=1.;fields['last_component']=last
            if len(coefs)!=(len(children) if a['extended'] else len(children)-1):raise UnsupportedModelError(f'{label}: invalid mixture layout')
            result=emit('mixture_yields' if a['extended'] else 'mixture_fractions',R,fields,children,coefs,label)
        elif cls=='RooProdPdf':
            factor=1.;children=[]
            if mode=='density' and axes:
                normalizations=product_normalizations(name,nodes[name],axes,depends)
                children=[request(channel_index,ch,n,'density',norm) for n,norm in zip(inputs,normalizations,strict=True) if norm]
            elif mode=='integral':
                used=set()
                for n in inputs:
                    selected=tuple(v for v in axes if depends(n,v))
                    if used&set(selected):raise UnsupportedModelError(f'{label}: overlapping raw product integral')
                    used.update(selected);children.append(request(channel_index,ch,n,'integral',selected))
                factor=math.prod(nodes[v][2]['maximum']-nodes[v][2]['minimum'] for v in set(axes)-used)
            else:children=[request(channel_index,ch,n,'value',()) for n in inputs]
            result=(emit('product',R,dict(child_mask=np.ones(len(children),dtype=bool),factor=np.array(factor)),children,label=label)
                if children else fixed(np.full(R,factor),label))
        elif cls=='VerticalInterpPdf':
            if mode=='density':
                children=[request(channel_index,ch,name,m,axes) for m in ('value','integral')]
                result=emit('divide',R,{},children,label=label)
            else:
                children=[request(channel_index,ch,n,mode,axes) for n in a['functions']]
                if len(children)!=2*len(a['coefs'])+1 or a['algorithm']<0 or a['region']<0:raise UnsupportedModelError(f'{label}: invalid vertical interpolation')
                fields=scalar_fields(a['coefs'],minimum=0)
                fields.update(region=np.array(a['region'],dtype=np.float64),pdf_floor=np.array(a['pdf_floor'],dtype=np.float64),
                    second_floor=np.array(a['integral_floor'] if mode=='integral' else a['pdf_floor'],dtype=np.float64))
                op=f'vertical_{a["algorithm"]}' if a['algorithm'] in (0,1) else 'vertical_polynomial'
                result=emit(op,R,fields,children,a['coefs'],label)
        elif cls in ('RooParametricHist2D','RooParametricStepFunction'):
            if mode=='density':
                children=[request(channel_index,ch,name,m,axes) for m in ('value','integral')]
                result=emit('divide',R,{},children,label=label)
            else:
                coefs=a['coefs'];fields=scalar_fields(coefs)
                if mode=='integral':
                    if set(axes)!=set(a['axes']):raise UnsupportedModelError(f'{label}: parametric template requires a full joint integral')
                    result=(fixed(np.ones(R),label) if cls=='RooParametricStepFunction'
                        else emit('coef_sum',R,{**fields,'broadcast':np.zeros(R)},coef_names=coefs,label=label))
                elif cls=='RooParametricStepFunction':
                    e=np.asarray(a['edges']);x=np.asarray(coords[a['axes'][0]])
                    if len(coefs)!=len(e)-2 or np.any(np.diff(e)<=0):raise UnsupportedModelError(f'{label}: invalid step geometry')
                    index=np.clip(np.searchsorted(e,x,side='right')-1,0,len(e)-2)
                    widths=np.zeros(len(fields['coefs']));widths[:len(coefs)]=np.diff(e)[:-1]
                    fields.update(data_bins=np.minimum(index,max(0,len(coefs)-1)).astype(np.int32),last_bin=index==len(coefs),
                        data_valid=(x>=e[0])&(x<e[-1]),widths=widths,last_width=np.array(e[-1]-e[-2],dtype=np.float64))
                    result=emit('step_bins',R,fields,coef_names=coefs,label=label)
                else:
                    indices=[];valid=np.ones(R,dtype=bool);volume=np.ones(R)
                    if len(coefs)!=math.prod(a['shape']) or len(a['axes'])!=2:raise UnsupportedModelError(f'{label}: invalid parametric grid')
                    for axis,e,width in zip(a['axes'],a['edges'],a['widths'],strict=True):
                        x=np.asarray(coords[axis]);idx=np.clip(np.searchsorted(e,x,side='right')-1,0,len(e)-2)
                        indices.append(idx);valid&=(x>=e[0])&(x<e[-1]);volume*=np.asarray(width)[idx]
                    fields.update(data_bins=(indices[1]*a['shape'][0]+indices[0]).astype(np.int32),data_valid=valid,volume=volume)
                    result=emit('parametric_bins',R,fields,coef_names=coefs,label=label)
        elif cls=='RooUniform':
            if mode!='value' and not set(axes)<=set(inputs):raise UnsupportedModelError(f'{label}: incompatible uniform normalization')
            volume=math.prod(nodes[v][2]['maximum']-nodes[v][2]['minimum'] for v in axes)
            result=fixed(np.full(R,1. if mode=='value' else volume if mode=='integral' else 1./volume),label)
        elif cls=='RooHistPdf':
            if mode!='value':
                if set(axes)!=set(a['axes']):raise UnsupportedModelError(f'{label}: partial histogram integral')
                for axis,e in zip(a['axes'],a['edges'],strict=True):
                    if nodes[axis][2]['minimum']>e[0] or nodes[axis][2]['maximum']<e[-1]:raise UnsupportedModelError(f'{label}: restricted histogram integral')
            if mode=='integral':result=fixed(np.full(R,a['sum_weights']),label)
            else:
                index=[];valid=np.ones(R,dtype=bool)
                for axis,e in zip(a['axes'],a['edges'],strict=True):
                    x=np.asarray(coords[axis]);index.append(np.clip(np.searchsorted(e,x,side='right')-1,0,len(e)-2));valid&=(x>=e[0])&(x<=e[-1])
                value=np.where(valid,np.maximum(np.asarray(a['values'])[tuple(index)],0.),0.)
                result=fixed(value if mode=='value' else value/a['sum_weights'],label)
        else:raise UnsupportedModelError(f'{label}: unsupported binned PDF {cls}')
        cache[key]=result;return result
    channel_records=[]
    def correction(name):
        cls,_,a=nodes[name]
        if cls=='RooMultiPdf':return float(a['correction'])
        if cls=='RooAddPdf':return sum(correction(n) for n in a['pdfs'])
        return 0.
    for i,ch in enumerate(channels):
        if ch.get('masked',False):
            channel_records.append(dict(name=ch['name'],masked=True));continue
        cls,_,a=nodes[ch['pdf']]
        if cls!='RooAddPdf' or not a['extended']:raise UnsupportedModelError(f'{ch["name"]}: requires extended RooAddPdf')
        weights=np.asarray(ch['weights'],dtype=np.float64)
        if weights.ndim!=1 or not np.all(np.isfinite(weights)) or np.any(weights<0):raise UnsupportedModelError(f'{ch["name"]}: invalid weights')
        if any(n not in ch['coordinates'] or np.shape(ch['coordinates'][n])!=weights.shape or not np.all(np.isfinite(ch['coordinates'][n])) for n in ch['observables']):
            raise UnsupportedModelError(f'{ch["name"]}: invalid data coordinates')
        density=request(i,ch,ch['pdf'],'density',ch['observables']);fields=scalar_fields(a['coefs'])
        fields=dict(densities=density[0],weights=weights,rates=fields['coefs'],rate_mask=fields['coef_mask'],correction=np.array(correction(ch['pdf'])))
        channel_records.append(dict(name=ch['name'],masked=False,fields=fields,dynamic=density[1] or any(n in dynamic for n in a['coefs'])))
    if len({r['name'] for r in channel_records})!=len(channel_records):raise ValueError('Duplicate channel name')
    queries=pack_fast_queries(histograms,fast_requests);query_lookup={q['name']:q for q in queries.metadata['queries']}
    fast_requests.clear();cache.clear();grids.clear()
    initial=np.concatenate(initial_chunks) if initial_chunks else np.empty(0);initial_chunks.clear()
    fast_outputs=[];fast_inputs=[]
    for output,label in fast_bindings:
        q=query_lookup[label];indices=np.arange(q['offset'],q['offset']+q['size'],dtype=np.int32)
        indices=np.broadcast_to(indices,output.shape)
        if label in queries.metadata['folded_queries']:initial[output]=queries.arrays['initial_values'][indices];folded.append(label)
        else:fast_outputs.append(output);fast_inputs.append(indices)
    groups=defaultdict(list)
    for r in records:
        if r['dynamic']:groups[(r['depth'],r['op'],r['rows'])].append(r)
        else:
            value=operation(r['op'],initial,scalar_initial,{k:v[None,...] for k,v in r['fields'].items()})[0]
            if not np.all(np.isfinite(value)):raise UnsupportedModelError(f'{r["label"]}: invalid fixed density')
            initial[r['outputs']]=value;folded.append(r['label'])
    records.clear()
    arrays=dict(initial_values=initial,fast_outputs=np.concatenate(fast_outputs).astype(np.int32) if fast_outputs else np.empty(0,dtype=np.int32),
        fast_inputs=np.concatenate(fast_inputs).astype(np.int32) if fast_inputs else np.empty(0,dtype=np.int32),**data_arrays)
    fast_outputs.clear();fast_inputs.clear();fast_bindings.clear()
    schedule=[]
    for gi,((depth,op,rows),group) in enumerate(sorted(groups.items())):
        fields=_stack_records([r['fields'] for r in group]);fields['outputs']=np.stack([r['outputs'] for r in group])
        bindings={k:f'program_{gi}_{k}' for k in fields};arrays.update({bindings[k]:v for k,v in fields.items()})
        schedule.append(dict(operation=op,depth=depth,buffers=bindings))
        group.clear()
    channel_initial=np.zeros(len(channel_records));channel_groups=defaultdict(list);folded_channels=[]
    for i,r in enumerate(channel_records):
        if r['masked']:folded_channels.append(r['name'])
        elif not r['dynamic']:
            value=nll_batch(initial,scalar_initial,{k:v[None,...] for k,v in r['fields'].items()})[0]
            if not np.isfinite(value):raise UnsupportedModelError(f'{r["name"]}: invalid fixed NLL')
            channel_initial[i]=value;folded_channels.append(r['name'])
        else:channel_groups[len(r['fields']['weights'])].append((i,r['fields']))
    arrays['initial_nll']=channel_initial;nll_schedule=[]
    for gi,(rows,group) in enumerate(sorted(channel_groups.items())):
        fields=_stack_records([r for _,r in group]);fields['outputs']=np.array([i for i,_ in group],dtype=np.int32)
        bindings={k:f'nll_{gi}_{k}' for k in fields};arrays.update({bindings[k]:v for k,v in fields.items()})
        nll_schedule.append(dict(operation='extended_binned_mixture',buffers=bindings))
    for key,bundle in [('histograms',histograms),('queries',queries)]:arrays.update({key+'__'+n:v for n,v in bundle.arrays.items()})
    result=TensorBundle(dict(schema='combine-jax.tensor.v1',kind='binned_channels',
        schedule=[dict(component='histograms',schedule=histograms.metadata['schedule']),dict(component='queries',schedule=queries.metadata['schedule']),
            dict(component='densities',schedule=schedule),dict(component='nll',schedule=nll_schedule)],
        histograms=histograms.metadata,queries=queries.metadata,program=schedule,nll_program=nll_schedule,
        scalar_names=scalars.metadata['output_names'],names=[r['name'] for r in channel_records],folded_queries=folded,folded_channels=folded_channels,
        analytic_grid_queries=grid_queries,compact_shape_data=bool(compact_data),data_layouts=data_layouts,
        active_pdf_classes=sorted({nodes[n][0] for n in active_pdfs}),
        scope='All supplied binned channels, with explicit normalization requests, fixed folding and batched density/NLL reductions.'),arrays)
    validate_binned_channels(result)
    return result


def unpack_binned(bundle):
    return {key:TensorBundle._from_frozen(bundle.metadata[key],{n[len(key)+2:]:a for n,a in bundle.arrays.items() if n.startswith(key+'__')}) for key in ('histograms','queries')}


def _claim_outputs(values,seen):
    """Certify disjoint scatter destinations without Python objects per index."""
    flat=values.reshape(-1)
    if np.any(seen[flat]) or np.unique(flat).size!=flat.size:
        raise ValueError('Repeated binned outputs')
    seen[flat]=True


def validate_binned_channels(bundle):
    m,a=bundle.metadata,bundle.arrays
    if m['kind']!='binned_channels':raise ValueError('Expected binned channels')
    validate_shape_data(m,a)
    parts=unpack_binned(bundle);validate_fast_histograms(parts['histograms']);validate_fast_queries(parts['queries'])
    if parts['queries'].metadata['histogram_structure_key']!=parts['histograms'].structure_key or parts['histograms'].metadata['scalar_names']!=m['scalar_names']:
        raise ValueError('Binned component bindings differ')
    expected=[dict(component='histograms',schedule=m['histograms']['schedule']),dict(component='queries',schedule=m['queries']['schedule']),
        dict(component='densities',schedule=m['program']),dict(component='nll',schedule=m['nll_program'])]
    if m['schedule']!=expected:raise ValueError('Binned program schedules differ')
    if len(m['names'])!=len(set(m['names'])) or a['initial_nll'].shape!=(len(m['names']),):raise ValueError('Invalid channel layout')
    for name in ('initial_values','initial_nll'):
        if a[name].ndim!=1 or a[name].dtype!=np.float64 or not np.all(np.isfinite(a[name])):raise ValueError(f'Invalid binned buffer {name}')
    N=len(a['initial_values']);S=len(m['scalar_names']);Q=len(parts['queries'].arrays['initial_values'])
    def indices(values,limit,label):
        if values.dtype!=np.int32 or np.any(values<0) or np.any(values>=limit):raise ValueError(f'Invalid binned index {label}')
    indices(a['fast_outputs'],N,'fast_outputs');indices(a['fast_inputs'],Q,'fast_inputs')
    if a['fast_outputs'].shape!=a['fast_inputs'].shape:raise ValueError('Fast output shape differs')
    outputs=np.zeros(N,dtype=bool);channel_outputs=np.zeros(len(m['names']),dtype=bool)
    _claim_outputs(a['fast_outputs'],outputs)
    for sequence,is_nll in ((m['program'],False),(m['nll_program'],True)):
        for batch in sequence:
            b={k:a[n] for k,n in batch['buffers'].items()}
            op=batch['operation']
            allowed=ANALYTIC_OPERATIONS|GRID_ANALYTIC_OPERATIONS|GRID_OPERATIONS|{'divide','product','coef_sum','parametric_bins','step_bins','mixture_yields','mixture_fractions',
                     'vertical_0','vertical_1','vertical_polynomial'}
            if op not in ({'extended_binned_mixture'} if is_nll else allowed):raise ValueError('Unknown binned operation')
            indices(b['outputs'],len(m['names']) if is_nll else N,'outputs')
            if b['outputs'].ndim!=(1 if is_nll else 2):raise ValueError('Invalid binned output rank')
            H=len(b['outputs']);R=b['weights'].shape[1] if is_nll else b['outputs'].shape[1]
            _claim_outputs(b['outputs'],channel_outputs if is_nll else outputs)
            for n in ('coefs','rates'):
                if n in b:
                    indices(b[n],S,n);mask=b['rate_mask' if n=='rates' else 'coef_mask']
                    if b[n].ndim!=2 or len(b[n])!=H or mask.dtype!=bool or mask.shape!=b[n].shape:raise ValueError('Invalid coefficient mask')
            for n in ('children','densities'):
                if n in b:
                    indices(b[n],N,n)
                    if b[n].ndim!=(3 if n=='children' else 2) or b[n].shape[0]!=H or b[n].shape[-1]!=R:raise ValueError('Invalid density gather shape')
            if 'child_mask' in b and (b['child_mask'].dtype!=bool or b['child_mask'].shape!=b['children'].shape[:2]):raise ValueError('Invalid child mask')
            if 'data_bins' in b:
                indices(b['data_bins'],b['coefs'].shape[1],'data_bins')
                if b['data_bins'].shape!=(H,R) or b['data_valid'].shape!=(H,R) or b['data_valid'].dtype!=bool:raise ValueError('Invalid data gather')
            if is_nll and (b['weights'].shape!=(H,R) or np.any(b['weights']<0) or b['correction'].shape!=(H,)):raise ValueError('Invalid extended NLL buffers')
            if op=='divide' and b['children'].shape[1]!=2:raise ValueError('Division needs two inputs')
            if op.startswith('mixture_') and b['children'].shape[1]!=b['coefs'].shape[1]:raise ValueError('Mixture component/rate mismatch')
            if op.startswith('vertical_') and b['children'].shape[1]!=2*b['coefs'].shape[1]+1:raise ValueError('Vertical component/morph mismatch')
            if op in ANALYTIC_OPERATIONS|GRID_ANALYTIC_OPERATIONS:validate_analytic_fields(op,b,R)
            if op in GRID_OPERATIONS:validate_grid_fields(op,b,R)
            for n in ('volume','last_width'):
                if n in b and np.any(b[n]<=0):raise ValueError('Invalid bin volume/width')
            for n,value in b.items():
                if value.dtype.kind=='f' and (value.dtype!=np.float64 or not np.all(np.isfinite(value))):raise ValueError(f'Invalid numerical binned buffer {n}')


def evaluate_binned_channels(bundle,scalar_values):
    validate_binned_channels(bundle)
    if np.shape(scalar_values)!=(len(bundle.metadata['scalar_names']),):raise ValueError('Scalar vector shape mismatch')
    parts=unpack_binned(bundle);a=bundle.arrays
    fast=evaluate_fast_queries(parts['queries'],evaluate_fast_histograms(parts['histograms'],scalar_values))
    values=a['initial_values'].copy();values[a['fast_outputs']]=fast[a['fast_inputs']]
    for batch in bundle.metadata['program']:
        b={k:a[n] for k,n in batch['buffers'].items()};values[b['outputs']]=operation(batch['operation'],values,scalar_values,b)
    output=a['initial_nll'].copy()
    for batch in bundle.metadata['nll_program']:
        b={k:a[n] for k,n in batch['buffers'].items()};output[b['outputs']]=nll_batch(values,scalar_values,b)
    return output


def make_binned_function(bundle, *, gradient='ad', row_chunk_size=1024):
    import jax
    import jax.numpy as jnp
    if gradient not in ('ad', 'structural', 'structural_sequential'):
        raise ValueError('Unknown binned gradient policy')
    validate_binned_channels(bundle);parts=unpack_binned(bundle)
    # Recompute histogram morph intermediates in reverse AD rather than retain
    # them alongside the much larger event-density tape. Runtime model arrays
    # are already resident inputs and do not need additional saved copies.
    hf=jax.checkpoint(make_fast_function(parts['histograms']))
    qf=make_query_function(parts['queries'])
    component_fields={k:tuple(v.arrays) for k,v in parts.items()}
    program=tuple(bundle.metadata['program']);nll_program=tuple(bundle.metadata['nll_program']);size=len(bundle.metadata['scalar_names'])
    def histogram_queries(scalar_values,arrays):
        components={k:{n:arrays[k+'__'+n] for n in names} for k,names in component_fields.items()}
        return qf(hf(scalar_values,components['histograms']),components['queries'])
    def densities(scalar_values,arrays):
        if scalar_values.shape!=(size,):raise ValueError('Scalar vector shape mismatch')
        fast=histogram_queries(scalar_values,arrays)
        # validate_binned_channels certifies unique destinations, including all
        # fast bindings and every program output. Supplying that guarantee avoids
        # scatter-set AD bookkeeping for potentially repeated destinations.
        selected=fast.at[arrays['fast_inputs']].get(mode='promise_in_bounds',wrap_negative_indices=False)
        values=arrays['initial_values'].at[arrays['fast_outputs']].set(selected,
            unique_indices=True,mode='promise_in_bounds',wrap_negative_indices=False)
        for batch in program:
            b={k:arrays[n] for k,n in batch['buffers'].items()};values=values.at[b['outputs']].set(operation(batch['operation'],values,scalar_values,b,jnp),
                unique_indices=True,mode='promise_in_bounds',wrap_negative_indices=False)
        return values
    def evaluate(scalar_values,arrays):
        values=densities(scalar_values,arrays)
        output=arrays['initial_nll']
        for batch in nll_program:
            b={k:arrays[n] for k,n in batch['buffers'].items()};output=output.at[b['outputs']].set(nll_batch(values,scalar_values,b,jnp),unique_indices=True)
        return output
    if gradient!='ad':
        from .structural_gradient import make_structural_pullback
        return make_structural_pullback(evaluate,densities,histogram_queries,program,nll_program,
                                        row_chunk_size=row_chunk_size,sequential_shapes=gradient=='structural_sequential')
    return evaluate
