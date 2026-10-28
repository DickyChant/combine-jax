"""Rederive HHVV row selection and compare full normalization/profile buffers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from combine_jax.tensor.bundle import TensorBundle


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_compaction(model,old_import_path):
    old_import=json.loads(old_import_path.read_bytes());assert old_import['status']=='passed'
    for path,digest in old_import['artifacts'].items():assert sha(path)==digest
    old=TensorBundle.load(old_import['model'])
    assert old.metadata['parameter_names']==model.metadata['parameter_names']
    assert old.metadata['channel_names']==model.metadata['channel_names']
    assert old.metadata['provenance']==model.metadata['provenance']
    prefixes=('offset','scalars__','constraints__','channels__','templates__','binned__histograms__')
    keys={n for n in old.arrays if n.startswith(prefixes)}
    assert keys=={n for n in model.arrays if n.startswith(prefixes)}
    for name in keys:np.testing.assert_array_equal(model.arrays[name],old.arrays[name])
    for key in ('scalars','constraints','channels','templates'):
        assert model.metadata['components'][key]==old.metadata['components'][key]
    assert model.metadata['components']['binned']['histograms']==old.metadata['components']['binned']['histograms']
    old_bytes=sum(a.nbytes for a in old.arrays.values());del old
    # This is the hash-bound, locally generated oracle export used by the census.
    scalar_path=Path('reports/tensor_hhvv_scalars.json');scalar=json.loads(scalar_path.read_bytes())
    export=Path(scalar['export']);assert sha(export)==scalar['export_sha256']
    with export.open('rb') as stream:payload=pickle.load(stream)
    rows={r[0]:r for r in payload['channels']};b=model.metadata['components']['binned']
    assert b['compact_shape_data'] is True
    totals=Counter();details=[];arrays=model.arrays
    for q in b['data_layouts']:
        name,pdf,axes,coords,weights,storage,masked=rows[q['name']]
        assert q['masked']==masked
        if masked:continue
        selected=np.flatnonzero(weights>0);count=len(selected);size=1<<(count-1).bit_length() if count else 0
        assert (q['original_rows'],q['positive_rows'],q['retained_rows'],q['packed_rows'])==(len(weights),count,count,size)
        indices=arrays['binned__'+q['source_indices']];mask=arrays['binned__'+q['source_mask']]
        expected=np.full(size,int(selected[0]) if count else 0,dtype=np.int32);expected[:count]=selected
        np.testing.assert_array_equal(indices,expected);np.testing.assert_array_equal(mask,np.arange(size)<count)
        assert q['observed_sum']==float(np.sum(weights))
        packed=np.where(mask,weights[indices],0.)
        assert abs(float(np.sum(packed))-q['observed_sum'])<1e-9
        position=b['names'].index(name);found=False
        for batch in b['nll_program']:
            outputs=arrays['binned__'+batch['buffers']['outputs']]
            matches=np.flatnonzero(outputs==position)
            if len(matches):
                assert len(matches)==1 and not found;found=True
                np.testing.assert_array_equal(arrays['binned__'+batch['buffers']['weights']][matches[0]],packed)
        assert found or name in b['folded_channels']
        detail={k:q[k] for k in ('name','original_rows','positive_rows','packed_rows')};details.append(detail)
        totals.update({k:detail[k] for k in ('original_rows','positive_rows','packed_rows')})
    grids=b['analytic_grid_queries']
    for q in grids:
        distinct=q['distinct_grid_rows'];assert q['grid_rows']==(1<<(distinct-1).bit_length() if distinct else 0)
    return dict(status='passed',script_sha256=sha(__file__),uncompacted_import=str(old_import_path),
        uncompacted_import_sha256=sha(old_import_path),export=str(export),export_sha256=sha(export),
        scalar_report=str(scalar_path),scalar_report_sha256=sha(scalar_path),
        unchanged_normalization_and_component_buffers=len(keys),
        unchanged_scope='All scalar, counting, CMS template/profile, constraint and full fast-histogram normalization arrays and component metadata; offset unchanged.',
        rows=dict(totals),channels=details,grid_queries=len(grids),
        grid_distinct_rows=sum(q['distinct_grid_rows'] for q in grids),grid_padded_rows=sum(q['grid_rows'] for q in grids),
        buffer_bytes_before=old_bytes,buffer_bytes_after=sum(a.nbytes for a in arrays.values()),
        policy='Only zero-weight rows of extended shape terms are removed. Power-of-two padding duplicates retained coordinates with zero weight; source selections are independently rederived.')
