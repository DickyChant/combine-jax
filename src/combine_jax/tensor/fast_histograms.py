"""Framework-neutral batches for Combine's self-normalized fast histogram PDFs.

Histograms retain their full normalization grids. Queries are a separate import
step: data gathers and exact geometric integrals never alter those grids.
"""
from collections import defaultdict
import math

import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle
from .scalar import validate_program


FAST_CLASSES=frozenset(('FastVerticalInterpHistPdf','FastVerticalInterpHistPdf2',
    'FastVerticalInterpHistPdf2D','FastVerticalInterpHistPdf2D2','FastVerticalInterpHistPdf3D'))
ONE_DIMENSIONAL=frozenset(('FastVerticalInterpHistPdf','FastVerticalInterpHistPdf2'))


def morph_weights(theta,region,xp=np):
    safe=xp.where(region>0,region,1.)[:,None]
    x=theta/safe
    polynomial=x*(x*x*(3*x*x-10)+15)/8
    return xp.where(xp.abs(theta)>=region[:,None],xp.where(theta>0,1.,-1.),polynomial)


def normalized_batch(logarithmic,theta,buffers,xp=np):
    """Two batched matrix products; no morph-by-bin intermediate is materialized."""
    theta=xp.where(buffers['morph_mask'],theta,0.)
    alpha=morph_weights(theta,buffers['region'],xp)
    values=(buffers['base']+xp.einsum('hm,hmsk->hsk',.5*theta,buffers['diffs'])
        +xp.einsum('hm,hmsk->hsk',.5*theta*alpha,buffers['sums']))
    values=xp.exp(values) if logarithmic else xp.maximum(values,1e-9)
    total=xp.sum(values*buffers['widths'],axis=2,keepdims=True)
    values=values/xp.where(total>0,total,1.)
    return xp.concatenate((values.reshape((values.shape[0],-1)),buffers['tail']),axis=1)


def scalar_bindings(scalars):
    validate_program(scalars)
    names=scalars.metadata['output_names'];lookup={n:i for i,n in enumerate(names)}
    slots=set(scalars.arrays['parameter_slots'].tolist())
    for batch in scalars.metadata['schedule']:slots.update(scalars.arrays[batch['outputs']].tolist())
    dynamic={n for n,s in zip(names,scalars.arrays['output_slots'],strict=True) if int(s) in slots}
    constant={n:float(scalars.arrays['initial_values'][s]) for n,s in zip(names,scalars.arrays['output_slots'],strict=True) if n not in dynamic}
    return lookup,dynamic,constant


def pack_fast_histograms(nodes,names,scalars,*,provenance=None):
    """Import certified old-extractor node records; no ROOT/device framework used.

    Observable-dependent morph coefficients are rejected through the scalar map.
    Fixed morph contributions and entire fixed histograms are precomputed here.
    Grid layouts, not the number of nodes, determine the runtime batch schedule.
    """
    lookup,dynamic,constant=scalar_bindings(scalars)
    names=list(names)
    if len(set(names))!=len(names):raise ValueError('Duplicate fast histogram name')
    arrays={};histograms=[];groups=defaultdict(list);initial=[];offset=0;folded=[]
    for index,name in enumerate(names):
        cls,_,a=nodes[name]
        if cls not in FAST_CLASSES:raise UnsupportedModelError(f'{name}: unsupported fast histogram class {cls}')
        one=cls in ONE_DIMENSIONAL
        axes=(a['observable_name'],) if one else tuple(a['axes'])
        edges=(np.asarray(a['edges']),) if one else tuple(np.asarray(e) for e in a['edges'])
        shape=tuple(len(e)-1 for e in edges);size=math.prod(shape)
        if len(axes)!=len(edges) or len(set(axes))!=len(axes) or not 1<=len(axes)<=3 or not size:
            raise UnsupportedModelError(f'{name}: invalid fast histogram axes')
        if any(e.ndim!=1 or not np.all(np.isfinite(e)) or not np.all(np.diff(e)>0) for e in edges):
            raise UnsupportedModelError(f'{name}: invalid fast histogram edges')
        if not one and tuple(a['shape'])!=shape:raise UnsupportedModelError(f'{name}: grid shape differs from edges')
        active=int(a['active']) if one else size
        conditional=False if one else bool(a['conditional'])
        slices=shape[0] if conditional else 1;bins=active//slices
        if not 0<active<=size or active%slices:raise UnsupportedModelError(f'{name}: invalid active grid')
        nominal=np.asarray(a['nominal'],dtype=np.float64)
        widths=np.asarray(a['widths'],dtype=np.float64)
        coefs=tuple(a['coefs']);region=float(a['region'])
        if nominal.shape!=shape or widths.size!=active or not math.isfinite(region) or region<0:
            raise UnsupportedModelError(f'{name}: invalid morph storage/region')
        if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(widths)) or np.any(widths<=0):
            raise UnsupportedModelError(f'{name}: nonfinite template or invalid normalization widths')
        base=nominal.reshape(-1)[:active].reshape(slices,bins).copy()
        diffs=np.asarray(a['diffs'],dtype=np.float64);sums=np.asarray(a['sums'],dtype=np.float64)
        if diffs.size!=len(coefs)*active or sums.size!=diffs.size or not np.all(np.isfinite(diffs)) or not np.all(np.isfinite(sums)):
            raise UnsupportedModelError(f'{name}: invalid morph tensors')
        diffs=diffs.reshape(len(coefs),slices,bins);sums=sums.reshape(diffs.shape)
        inputs=[];keep=[]
        for i,n in enumerate(coefs):
            if n not in lookup:raise UnsupportedModelError(f'{name}: missing independent scalar morph {n}')
            if n in dynamic:inputs.append(lookup[n]);keep.append(i)
            else:
                theta=constant[n];alpha=morph_weights(np.array([[theta]]),np.array([region]))[0,0]
                base+=.5*theta*(diffs[i]+alpha*sums[i])
        tail=np.asarray(a['tail'],dtype=np.float64) if one else np.empty(0)
        if tail.shape!=(size-active,) or not np.all(np.isfinite(tail)):raise UnsupportedModelError(f'{name}: invalid stored tail')
        bounds=[];edge_names=[]
        for d,(axis,edge) in enumerate(zip(axes,edges,strict=True)):
            cls_axis,_,aa=nodes[axis]
            if cls_axis!='RooRealVar':raise UnsupportedModelError(f'{name}: non-direct observable {axis}')
            lo,hi=aa['minimum'],aa['maximum']
            if not math.isfinite(lo) or not math.isfinite(hi) or not lo<hi:raise UnsupportedModelError(f'{name}: invalid observable bounds')
            bounds.append((lo,hi));key=f'grid_{index}_edges_{d}';edge_names.append(key);arrays[key]=edge
        arrays[f'grid_{index}_bounds']=np.array(bounds,dtype=np.float64)
        histograms.append(dict(name=name,class_name=cls,axes=axes,shape=shape,edges=edge_names,bounds=f'grid_{index}_bounds',
            offset=offset,size=size,active=active,conditional=conditional))
        record=dict(index=index,base=base,inputs=inputs,diffs=diffs[keep],sums=sums[keep],region=region,
            widths=widths.reshape(slices,bins),tail=tail,outputs=np.arange(offset,offset+size,dtype=np.int32))
        values=np.zeros(size)
        if not inputs:
            b={k:record[k][None,...] for k in ('base','diffs','sums','widths','tail')}
            b.update(region=np.array([region]),morph_mask=np.empty((1,0),dtype=bool))
            values=normalized_batch(bool(a['logarithmic']),np.empty((1,0)),b)[0]
            if not np.all(np.isfinite(values)):raise UnsupportedModelError(f'{name}: nonfinite fixed histogram')
            folded.append(name)
        else:groups[(bool(a['logarithmic']),slices,bins,size-active)].append(record)
        initial.extend(values);offset+=size
    arrays['initial_bins']=np.asarray(initial,dtype=np.float64);schedule=[]
    for gi,(key,records) in enumerate(sorted(groups.items())):
        logarithmic,S,K,T=key;H=len(records);M=max(len(r['inputs']) for r in records)
        fields=dict(base=np.zeros((H,S,K)),diffs=np.zeros((H,M,S,K)),sums=np.zeros((H,M,S,K)),
            widths=np.zeros((H,S,K)),tail=np.zeros((H,T)),region=np.zeros(H),morph_mask=np.zeros((H,M),dtype=bool),
            inputs=np.zeros((H,M),dtype=np.int32),outputs=np.zeros((H,S*K+T),dtype=np.int32))
        for i,r in enumerate(records):
            m=len(r['inputs']);fields['inputs'][i,:m]=r['inputs'];fields['morph_mask'][i,:m]=True
            fields['diffs'][i,:m]=r['diffs'];fields['sums'][i,:m]=r['sums']
            for n in ('base','widths','tail','region','outputs'):fields[n][i]=r[n]
        bindings={n:f'batch_{gi}_{n}' for n in fields};arrays.update({bindings[n]:v for n,v in fields.items()})
        schedule.append(dict(operation='normalized_fast_histograms',logarithmic=logarithmic,buffers=bindings))
    result=TensorBundle(dict(schema='combine-jax.tensor.v1',kind='fast_histograms',schedule=schedule,names=names,
        scalar_names=scalars.metadata['output_names'],histograms=histograms,folded_histograms=folded,provenance=provenance or {},
        scope='Full native normalization grids and stored tails; no sampled approximation or channel NLL implied.'),arrays)
    validate_fast_histograms(result)
    return result


def validate_fast_histograms(bundle):
    m,a=bundle.metadata,bundle.arrays
    if m['kind']!='fast_histograms':raise ValueError('Expected fast histogram bundle')
    if len(m['names'])!=len(set(m['names'])) or m['names']!=[h['name'] for h in m['histograms']]:raise ValueError('Invalid histogram names')
    if a['initial_bins'].ndim!=1 or a['initial_bins'].dtype!=np.float64 or not np.all(np.isfinite(a['initial_bins'])):raise ValueError('Invalid initial bin buffer')
    offset=0;fixed=set(m['folded_histograms']);expected=[]
    for h in m['histograms']:
        if h['offset']!=offset or h['size']!=math.prod(h['shape']) or len(h['axes'])!=len(h['shape']):raise ValueError('Invalid histogram layout')
        if not 0<h['active']<=h['size'] or h['class_name'] not in FAST_CLASSES:raise ValueError('Invalid histogram geometry/class')
        for size,key in zip(h['shape'],h['edges'],strict=True):
            if a[key].shape!=(size+1,) or not np.all(np.diff(a[key])>0):raise ValueError('Invalid edge buffer')
        if a[h['bounds']].shape!=(len(h['shape']),2):raise ValueError('Invalid bound buffer')
        if h['name'] not in fixed:expected.extend(range(offset,offset+h['size']))
        offset+=h['size']
    if offset!=len(a['initial_bins']) or not fixed<=set(m['names']):raise ValueError('Histogram coverage mismatch')
    positions=[]
    for batch in m['schedule']:
        if batch['operation']!='normalized_fast_histograms' or not isinstance(batch['logarithmic'],bool):raise ValueError('Unknown fast histogram operation')
        b={k:a[n] for k,n in batch['buffers'].items()}
        H,S,K=b['base'].shape;M=b['inputs'].shape[1];T=b['tail'].shape[1]
        shapes=dict(base=(H,S,K),diffs=(H,M,S,K),sums=(H,M,S,K),widths=(H,S,K),tail=(H,T),region=(H,),
            morph_mask=(H,M),inputs=(H,M),outputs=(H,S*K+T))
        if set(b)!=set(shapes):raise ValueError('Invalid morph buffer names')
        for n,shape in shapes.items():
            dtype=bool if n=='morph_mask' else np.int32 if n in ('inputs','outputs') else np.float64
            if b[n].shape!=shape or b[n].dtype!=dtype or not np.all(np.isfinite(b[n])):raise ValueError(f'Invalid fast histogram buffer {n}')
        if np.any(b['inputs']<0) or np.any(b['inputs']>=len(m['scalar_names'])):raise ValueError('Invalid scalar gather')
        if np.any(b['region']<0) or np.any(b['widths']<=0):raise ValueError('Invalid morph normalization/smoothing')
        positions.extend(b['outputs'].reshape(-1).tolist())
    if sorted(positions)!=expected:raise ValueError('Dynamic histogram output coverage differs')


def evaluate_fast_histograms(bundle,scalar_values):
    validate_fast_histograms(bundle)
    if np.shape(scalar_values)!=(len(bundle.metadata['scalar_names']),):raise ValueError('Scalar vector shape mismatch')
    output=bundle.arrays['initial_bins'].copy()
    for batch in bundle.metadata['schedule']:
        b={k:bundle.arrays[n] for k,n in batch['buffers'].items()}
        output[b['outputs']]=normalized_batch(batch['logarithmic'],np.asarray(scalar_values)[b['inputs']],b)
    return output


def make_fast_function(bundle):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    validate_fast_histograms(bundle)
    schedule=tuple(bundle.metadata['schedule']);size=len(bundle.metadata['scalar_names'])
    def evaluate(scalar_values,arrays):
        if scalar_values.shape!=(size,):raise ValueError('Scalar vector shape mismatch')
        output=arrays['initial_bins']
        for batch in schedule:
            b={k:arrays[n] for k,n in batch['buffers'].items()}
            # Validation proves disjoint output coverage. AD can therefore
            # omit last-writer bookkeeping for potentially repeated indices.
            output=output.at[b['outputs']].set(normalized_batch(batch['logarithmic'],scalar_values[b['inputs']],b,jnp),
                unique_indices=True,mode='promise_in_bounds',wrap_negative_indices=False)
        return output
    return evaluate


def pack_fast_queries(histograms,requests):
    """Compile point lookups and exact full/partial integrals into index maps.

    Requests use {name, pdf, mode, coordinates, axes}. Modes value/density retain
    the native selfNormalized convention; integral uses geometric overlaps with
    imported default bounds. Coordinates broadcast to a scalar or vector.
    """
    validate_fast_histograms(histograms)
    lookup={h['name']:h for h in histograms.metadata['histograms']};fixed=set(histograms.metadata['folded_histograms'])
    initial=[];gather=[];positions=[];validity=[];integral_indices=[];integral_outputs=[];integral_weights=[];records=[];folded=[];output_size=0
    for request in requests:
        label=request['name'];name=request['pdf'];mode=request.get('mode','density')
        if name not in lookup:raise UnsupportedModelError(f'{label}: histogram not imported: {name}')
        h=lookup[name];axes=tuple(request.get('axes',h['axes']));coordinates=request.get('coordinates',{})
        if len(set(axes))!=len(axes) or not set(axes)<=set(h['axes']):raise UnsupportedModelError(f'{label}: unsupported normalization axes')
        if mode not in ('density','value','integral'):raise UnsupportedModelError(f'{label}: unknown query mode {mode}')
        if mode=='density' and h['class_name'] in ONE_DIMENSIONAL and tuple(axes)!=tuple(h['axes']):raise UnsupportedModelError(f'{label}: 1D fast PDF requires its observable normalization set')
        integrate=set(axes) if mode=='integral' else set()
        remaining=[n for n in h['axes'] if n not in integrate]
        if not set(remaining)<=set(coordinates):raise UnsupportedModelError(f'{label}: missing query coordinates')
        values=np.broadcast_arrays(*(np.asarray(coordinates[n],dtype=np.float64) for n in remaining)) if remaining else [np.asarray(0.)]
        shape=values[0].shape
        if len(shape)>1 or any(not np.all(np.isfinite(v)) for v in values):raise UnsupportedModelError(f'{label}: invalid query coordinates')
        rows=int(np.prod(shape));coords={n:v.reshape(-1) for n,v in zip(remaining,values)}
        offset=output_size;output_size+=rows;initial_chunk=np.zeros(rows);initial.append(initial_chunk)
        records.append(dict(name=label,pdf=name,mode=mode,axes=axes,shape=shape,offset=offset,size=rows))
        edges=[histograms.arrays[k] for k in h['edges']];bounds=histograms.arrays[h['bounds']]
        if not integrate:
            indices=[];valid=np.ones(rows,dtype=bool)
            for n,e in zip(h['axes'],edges,strict=True):
                idx=np.searchsorted(e,coords[n],side='left')-1;valid&=(idx>=0)&(idx<len(e)-1)
                indices.append(np.clip(idx,0,len(e)-2))
            index=np.ravel_multi_index(tuple(indices),h['shape'])+h['offset']
            if name in fixed:
                initial_chunk[:]=np.where(valid,histograms.arrays['initial_bins'][index],0.);folded.append(label)
            else:
                gather.append(index.astype(np.int32));positions.append(np.arange(offset,offset+rows,dtype=np.int32));validity.append(valid)
        else:
            overlaps=[np.maximum(0.,np.minimum(e[1:],hi)-np.maximum(e[:-1],lo)) for e,(lo,hi) in zip(edges,bounds,strict=True)]
            for row in range(rows):
                indices=[];weights=[];valid=True
                for n,e,w in zip(h['axes'],edges,overlaps,strict=True):
                    if n in integrate:
                        keep=np.flatnonzero(w>0);indices.append(keep);weights.append(w[keep])
                    else:
                        i=int(np.searchsorted(e,coords[n][row],side='left'))-1;valid&=0<=i<len(e)-1
                        indices.append([np.clip(i,0,len(e)-2)]);weights.append([1.])
                if not valid:continue
                cells=np.meshgrid(*indices,indexing='ij');ws=np.meshgrid(*weights,indexing='ij')
                index=np.ravel_multi_index(tuple(c.reshape(-1) for c in cells),h['shape'])+h['offset']
                weight=np.prod(np.array(ws),axis=0).reshape(-1)
                if name in fixed:initial_chunk[row]=float(np.sum(histograms.arrays['initial_bins'][index]*weight))
                else:
                    integral_indices.append(index.astype(np.int32));integral_outputs.append(np.full(len(index),offset+row,dtype=np.int32));integral_weights.append(weight)
            if name in fixed:folded.append(label)
    if len({r['name'] for r in records})!=len(records):raise ValueError('Duplicate query name')
    def concatenate(chunks,dtype):
        result=np.concatenate(chunks,dtype=dtype) if chunks else np.empty(0,dtype=dtype)
        chunks.clear()
        return result
    arrays=dict(initial_values=concatenate(initial,np.float64),gather_indices=concatenate(gather,np.int32),
        gather_outputs=concatenate(positions,np.int32),gather_valid=concatenate(validity,bool),
        integral_indices=concatenate(integral_indices,np.int32),integral_outputs=concatenate(integral_outputs,np.int32),
        integral_weights=concatenate(integral_weights,np.float64))
    result=TensorBundle(dict(schema='combine-jax.tensor.v1',kind='fast_histogram_queries',
        schedule=[dict(operation='gather_and_integrate_fast_histograms')],queries=records,folded_queries=folded,
        histogram_structure_key=histograms.structure_key,histogram_names=histograms.metadata['names'],
        histogram_bins=len(histograms.arrays['initial_bins'])),arrays)
    validate_fast_queries(result)
    return result


def validate_fast_queries(bundle):
    m,a=bundle.metadata,bundle.arrays
    if m['kind']!='fast_histogram_queries' or m['schedule']!=[dict(operation='gather_and_integrate_fast_histograms')]:raise ValueError('Invalid fast query program')
    size=len(a['initial_values']);offset=0
    for q in m['queries']:
        if q['offset']!=offset or q['size']!=math.prod(q['shape']):raise ValueError('Invalid query output layout')
        offset+=q['size']
    if offset!=size or len({q['name'] for q in m['queries']})!=len(m['queries']):raise ValueError('Invalid query layout')
    for n,limit in [('gather_indices',m['histogram_bins']),('integral_indices',m['histogram_bins']),('gather_outputs',size),('integral_outputs',size)]:
        if a[n].ndim!=1 or a[n].dtype!=np.int32 or np.any(a[n]<0) or np.any(a[n]>=limit):raise ValueError(f'Invalid query gather {n}')
    if a['gather_indices'].shape!=a['gather_outputs'].shape or a['gather_valid'].shape!=a['gather_indices'].shape or a['gather_valid'].dtype!=bool:raise ValueError('Invalid query mask')
    if a['integral_indices'].shape!=a['integral_outputs'].shape or a['integral_weights'].shape!=a['integral_indices'].shape:raise ValueError('Invalid integral layout')
    for n in ('initial_values','integral_weights'):
        if a[n].ndim!=1 or a[n].dtype!=np.float64 or not np.all(np.isfinite(a[n])):raise ValueError('Invalid query numeric buffer')
    if np.any(a['integral_weights']<0):raise ValueError('Negative integration weight')
    seen=np.zeros(size,dtype=bool)
    if np.unique(a['gather_outputs']).size!=a['gather_outputs'].size:raise ValueError('Overlapping query outputs')
    seen[a['gather_outputs']]=True
    if np.any(seen[a['integral_outputs']]):raise ValueError('Overlapping query outputs')


def evaluate_fast_queries(bundle,bins):
    validate_fast_queries(bundle)
    if np.shape(bins)!=(bundle.metadata['histogram_bins'],):raise ValueError('Histogram bin vector shape mismatch')
    a=bundle.arrays;output=a['initial_values'].copy()
    output[a['gather_outputs']]=np.where(a['gather_valid'],np.asarray(bins)[a['gather_indices']],0.)
    np.add.at(output,a['integral_outputs'],np.asarray(bins)[a['integral_indices']]*a['integral_weights'])
    return output


def make_query_function(bundle):
    import jax.numpy as jnp
    validate_fast_queries(bundle);size=bundle.metadata['histogram_bins']
    def evaluate(bins,arrays):
        if bins.shape!=(size,):raise ValueError('Histogram bin vector shape mismatch')
        # Query destinations are unique; histogram source indices may repeat.
        # Integral destinations below deliberately accumulate repeated indices.
        selected=bins.at[arrays['gather_indices']].get(mode='promise_in_bounds',wrap_negative_indices=False)
        output=arrays['initial_values'].at[arrays['gather_outputs']].set(jnp.where(arrays['gather_valid'],selected,0.),
            unique_indices=True,mode='promise_in_bounds',wrap_negative_indices=False)
        integrated=bins.at[arrays['integral_indices']].get(mode='promise_in_bounds',wrap_negative_indices=False)
        return output.at[arrays['integral_outputs']].add(integrated*arrays['integral_weights'],
            mode='promise_in_bounds',wrap_negative_indices=False)
    return evaluate
