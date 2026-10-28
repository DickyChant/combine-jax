"""Apply unchanged native NLL/FD gates to fresh Asimov tensor captures."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tensor_benchmark_common import save_json,sha


def compare_coordinate(actual,case):
    """Never select a native reference using the AD value being tested."""
    status=case['status']
    if status=='fixed_bounds':
        return dict(status='fixed_bounds' if actual==0. else 'failed_fixed_derivative')
    if status!='converged':
        return dict(status='uncertified_native_'+status)
    first=case['selected_first_probe_index']
    count=case.get('reference_probe_count',3)
    assert count in (3,6)
    selected=case['probes'][first:first+count]
    assert len(selected)==count and all(p['status']=='valid' for p in selected)
    errors=[abs(actual-p['derivative'])/max(abs(p['derivative']),1e-8) for p in selected]
    passed=bool(np.isfinite(actual) and max(errors)<1e-4)
    return dict(status='passed' if passed else 'failed_gradient',max_relative_error=max(errors),
        max_absolute_error=max(abs(actual-p['derivative']) for p in selected))


def verified_refinements(paths,native,names,points,values):
    from native_fd_refinement import REFINEMENT_FRACTIONS,REFERENCE_SPREAD,select_reference
    selected={}
    for path in paths:
        report=json.loads(path.read_text())
        report_hash=sha(path)
        assert report['status']=='refinement_complete_pending_audit' and report['root_only']
        assert report['dataset']=='asimov_prefit'
        assert report['workspace_sha256']==native['workspace_sha256']
        assert report['generation_report_sha256']==native['generation_report_sha256']
        assert report['native_values_sha256']==native['nll_arrays_sha256']
        assert report['root_version']=='6.40.00' and report['combine_version']=='11.0.0'
        assert report['script_sha256']==sha('scripts/refine_asimov_native_reference.py')
        assert report['helper_sha256']==sha('scripts/native_fd_refinement.py')
        assert report['base_fd_sha256']==sha('scripts/native_fd_scan.py')
        assert report['reference_spread']==REFERENCE_SPREAD
        assert report['fractions']==list(REFINEMENT_FRACTIONS)
        for row in report['cases']:
            i,j=row['point'],row['index'];key=(i,j)
            assert key not in selected and 0<=i<len(points) and 0<=j<len(names)
            assert row['name']==names[j] and row['reference']['coordinate']==points[i,j]
            assert row['canonical_reference_error']<1e-6 and abs(row['canonical_nll']-values[i])<1e-6
            case=row['reference']
            assert [p['fraction'] for p in case['probes']]==list(REFINEMENT_FRACTIONS)
            choice=select_reference(case['probes'])
            assert all(case[k]==v for k,v in choice.items()), 'Native selection must reproduce without AD'
            selected[key]=dict(case=case,source=str(path),source_sha256=report_hash)
    return selected


def verified_cached_deltas(paths,native,names,points,values,*,centered=False):
    """Certify fixed-grid references formed before comparison with AD.

    The legacy small-derivative veto is a warning about cancellation in native
    NLL subtraction. It is not applied to the independently checked cache-delta
    arithmetic. Prefer an independently converged six-step window when available;
    otherwise retain the original three-step cache-delta policy. Histogram-score
    references require six steps. No AD value enters either selection. The
    full-spread selection and AD relative tolerances stay fixed.
    """
    from native_fd_refinement import REFINEMENT_FRACTIONS,select_reference
    selected={}
    for path in paths:
        report=json.loads(path.read_text())
        report_hash=sha(path)
        assert report['status']=='diagnostic_complete_pending_audit' and report['root_only']
        assert report['dataset']=='asimov_prefit'
        assert report['workspace_sha256']==native['workspace_sha256']
        assert report['generation_report_sha256']==native['generation_report_sha256']
        assert report['native_values_sha256']==native['nll_arrays_sha256']
        assert report['root_version']=='6.40.00' and report['combine_version']=='11.0.0'
        assert report['longdouble_mantissa_bits']>=63 and report['max_delta_parity_error']<1e-6
        if centered:
            assert report['reference_method']=='native_centered_histogram_score'
            assert report['runtime_flags']==native['runtime_flags']
        artifacts={str(Path(p).relative_to(report['workdir'])) if Path(p).is_absolute() else p:digest
                   for p,digest in report['artifacts'].items()}
        assert len(artifacts)==len(report['artifacts'])
        producer='scripts/probe_asimov_centered_hist_score.py' if centered else 'scripts/probe_asimov_cached_deltas.py'
        scripts=[producer,'scripts/native_cached_deltas.py','scripts/native_fd_refinement.py','scripts/native_fd_scan.py']
        if centered:scripts.append('scripts/native_centered_hist_score.py')
        for script in scripts:
            assert artifacts[script]==sha(script)
        assert report['fractions']==list(REFINEMENT_FRACTIONS)
        for row in report['cases']:
            i,j=row['point'],row['index'];key=(i,j)
            assert key not in selected and 0<=i<len(points) and 0<=j<len(names)
            assert row['name']==names[j] and row['reference']['coordinate']==points[i,j]
            assert row['canonical_reference_error']<1e-6 and abs(row['canonical_nll']-values[i])<1e-6
            case=row['reference']
            if centered and case['status']=='unsupported':
                assert not case['probes'] and row['reason']
                continue  # Keep the original uncertified reference; never waive it.
            assert row['max_delta_parity_error']<1e-6
            if centered:
                assert row['normalized_components']
                for component in row['normalized_components']:
                    assert component['class_name']=='FastVerticalInterpHistPdf2D2'
                    assert component['full_grid_rows']>0 and abs(component['integral']-1)<1e-12
            assert [p['fraction'] for p in case['probes']]==list(REFINEMENT_FRACTIONS)
            choice=select_reference(case['probes'])
            assert all(case[k]==v for k,v in choice.items()), 'Native selection must reproduce without AD'
            # Explicitly retain the original status and method with each override.
            certified={**case}
            if choice['status']=='roundoff_sensitive':certified['status']='converged'
            unconfirmed=certified
            from native_fd_confirmation import select_confirmed_reference
            confirmation=select_confirmed_reference(case['probes'])
            if centered or confirmation['status']!='unresolved':
                certified={k:v for k,v in case.items() if k not in
                    ('status','selected_first_probe_index','full_relative_spread')}
                certified.update(confirmation)
                if confirmation['status']=='roundoff_sensitive':certified['status']='converged'
            selected[key]=dict(case=certified,source=str(path),source_sha256=report_hash,
                reference_method='native_centered_histogram_score' if centered else 'native_cached_longdouble_delta',
                original_reference_status=case['status'])
            selected[key].update(unconfirmed_case=unconfirmed,confirmation=confirmation)
    return selected


def add_centered_references(references,centered):
    """Keep the unresolved predecessor; do not overwrite an accepted reference."""
    for key,replacement in centered.items():
        previous=references[key]
        assert previous.get('reference_method')=='native_cached_longdouble_delta'
        assert previous['case']['status']=='unresolved'
        references[key]={**replacement,'previous_reference':previous}


def native_point_records(native,shard_paths):
    """Merge disjoint native subsets; duplicates cannot count as new points."""
    from native_fd_scan import FRACTIONS
    records=list(native['points'])
    ids=[p['index'] for p in records]
    assert len(ids)==len(set(ids)) and all(0<=i<26 for i in ids)
    complete=True
    for path in shard_paths:
        document=json.loads(path.read_text())
        report_hash=sha(path)
        sealed=document['status']=='sealed_native_partition'
        if sealed:
            assert document['dataset']=='asimov_prefit' and document['worker_verified_stopped']
            assert document['signal']=='SIGTERM' and document['stopped_unix']>=document['started_unix']
            assert document['script_sha256']==sha('scripts/seal_asimov_native_partition.py')
            source=Path(document['source_report'])
            assert sha(source)==document['source_report_sha256']
            shard=json.loads(source.read_text())
            assert shard['workdir']==document['worker']['workdir']
            owned=document['owned_indices']
            assert owned and owned==sorted(set(owned)) and shard['selected_indices'][:len(owned)]==owned
            assert [p['index'] for p in shard['points']]==owned
        else:shard=document
        assert shard['status'] in ('running','shard_complete_pending_audit')
        assert shard['dataset']=='asimov_prefit'
        assert shard['workspace_sha256']==native['workspace_sha256']
        assert shard['generation_report_sha256']==native['generation_report_sha256']
        assert shard['native_values_sha256']==native['nll_arrays_sha256']
        assert shard['root_version']=='6.40.00' and shard['combine_version']=='11.0.0'
        assert shard['runtime_flags']==native['runtime_flags']
        assert shard['script_sha256']==sha('scripts/capture_asimov_native_fd_shard.py')
        assert shard['fd_script_sha256']==sha('scripts/native_fd_scan.py')
        assert shard['value_helper_sha256']==sha('scripts/capture_asimov_native_reference.py')
        assert shard['fractions']==list(FRACTIONS)
        planned=shard['selected_indices'];received=[p['index'] for p in shard['points']]
        assert planned==sorted(set(planned)) and all(0<=i<26 for i in planned)
        assert received==planned[:len(received)]
        selected=owned if sealed else planned
        assert not set(selected)&set(ids), 'Overlapping native point assignments'
        if shard['status']=='shard_complete_pending_audit':
            assert received==selected and shard['root_only']
        elif not sealed:complete=False
        records.extend({**p, 'reference_report':str(path),'reference_report_sha256':report_hash} for p in shard['points'])
        ids.extend(selected)
    return sorted(records,key=lambda p:p['index']),complete


def checked_native_point(record,names,points,values):
    artifact=Path(record['artifact']);assert sha(artifact)==record['artifact_sha256']
    raw=json.loads(gzip.decompress(artifact.read_bytes()))
    i=record['index'];assert raw['index']==i and len(raw['cases'])==len(names)
    assert abs(raw['canonical_nll']-values[i])<1e-6 and abs(raw['zero_point_nll'])<1e-6
    assert raw['restoration_error']<1e-6 and record['restoration_error']<1e-6
    for j,case in enumerate(raw['cases']):
        assert case['index']==j and case['name']==names[j] and case['coordinate']==points[i,j]
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','native','capture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--refinement',type=Path,action='append',default=[])
    parser.add_argument('--cached-deltas',type=Path,action='append',default=[])
    parser.add_argument('--centered-scores',type=Path,action='append',default=[])
    parser.add_argument('--native-shard',type=Path,action='append',default=[])
    args=parser.parse_args();assert not args.output.exists()
    from combine_jax.tensor.bundle import json_hash
    from tensor_benchmark_dataset import benchmark_dataset
    case=json.loads(args.manifest.read_text())['hhvv']
    # Metadata and archive identity suffice; avoid loading 2+ GiB of buffers.
    document=json.loads((Path(case['model'])/'model.json').read_text())
    assert json_hash(document['metadata'])==document['metadata_sha256']
    assert sha(Path(case['model'])/'arrays.npz')==document['arrays_sha256']
    benchmark_dataset(case,SimpleNamespace(metadata=document['metadata']),label='hhvv')
    native=json.loads(args.native.read_text());capture=json.loads(args.capture.read_text())
    assert native['dataset']=='asimov_prefit'
    assert native['workspace_sha256']==sha(case['workspace'])
    assert native['generation_report_sha256']==sha(case['generation_report'])
    assert native['script_sha256']==sha('scripts/capture_asimov_native_reference.py')
    assert native['fd_script_sha256']==sha('scripts/native_fd_scan.py')
    assert capture['status']=='measured_pending_validation' and capture['operation']=='value_gradient'
    assert capture['float64'] and not capture['root_imported']
    assert capture['model_sha256']=={n:sha(Path(case['model'])/n) for n in ('model.json','arrays.npz')}
    assert sha(capture['arrays'])==capture['arrays_sha256']
    assert sha(native['nll_arrays'])==native['nll_arrays_sha256']
    with np.load(capture['arrays'],allow_pickle=False) as saved:
        names=saved['parameter_names'];points=saved['points'];values=saved['values'];gradients=saved['gradients']
    with np.load(native['nll_arrays'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(names,saved['parameter_names'])
        np.testing.assert_array_equal(points,saved['points'])
        expected_values=saved['values']
    assert values.shape==(26,) and gradients.shape==points.shape==(26,len(names))
    refinements=verified_refinements(args.refinement,native,names,points,expected_values)
    cached=verified_cached_deltas(args.cached_deltas,native,names,points,expected_values)
    assert not set(cached)&set(refinements), 'Overlapping reference replacements'
    refinements.update(cached)
    centered=verified_cached_deltas(args.centered_scores,native,names,points,expected_values,centered=True)
    add_centered_references(refinements,centered)
    native_records,shards_complete=native_point_records(native,args.native_shard)
    report=dict(status='checking',dataset='asimov_prefit',script_sha256=sha(__file__),
        workspace_sha256=sha(case['workspace']),model_sha256=capture['model_sha256'],
        backend=capture['backend'],generation_report_sha256=sha(case['generation_report']),
        artifacts={str(p):sha(p) for p in (args.manifest,args.native,args.capture,*args.native_shard,
            *args.refinement,*args.cached_deltas,*args.centered_scores)},
        max_nll_absolute_error=float(np.max(abs(values-expected_values))),points=[],refinements_applied=[],
        policy='NLL absolute <1e-6. Gradient relative <1e-4 against every independently selected native estimate (three normally; six preferred when available for cache deltas and required for histogram scores), denominator max(abs(native),1e-8). Tiny/unresolved native derivatives are not silently waived. No fitted-data reference is loaded.')
    if args.cached_deltas:
        report['cached_delta_confirmation']=dict(helper_sha256=sha('scripts/native_fd_confirmation.py'),
            policy='Apply to every cache-delta case without AD input: prefer the first six-step window with full spread <1e-5; if none exists retain the original three-step native policy. Preserve both comparisons. This does not certify any originally unresolved reference without a valid window.')
    if args.centered_scores:
        from native_fd_confirmation import CONFIRMATION_PROBES
        report['centered_score_confirmation']=dict(required_probes=CONFIRMATION_PROBES,
            helper_sha256=sha('scripts/native_fd_confirmation.py'),
            policy='First six consecutive valid, distinct descending steps, same stencil/direction, full spread <1e-5. Applied to all histogram-score references without AD-dependent selection.')
    report['nll_passed']=bool(np.isfinite(values).all() and report['max_nll_absolute_error']<1e-6)
    counts=Counter()
    for point in native_records:
        raw=checked_native_point(point,names,points,expected_values)
        i=point['index']
        results=[]
        for j,reference in enumerate(raw['cases']):
            assert reference['index']==j and reference['name']==names[j]
            if (i,j) in refinements:
                replacement=refinements[(i,j)]
                evidence=dict(point=i,index=j,name=str(names[j]),
                    source=replacement['source'],source_sha256=replacement['source_sha256'],
                    reference_method=replacement.get('reference_method','native_zero_point_subtraction'),
                    original_reference_status=replacement.get('original_reference_status',replacement['case']['status']),
                    original_comparison=compare_coordinate(gradients[i,j],reference),
                    refined_comparison=compare_coordinate(gradients[i,j],replacement['case']))
                if 'previous_reference' in replacement:
                    previous=replacement['previous_reference']
                    evidence['previous_native_reference']=dict(source=previous['source'],
                        source_sha256=previous['source_sha256'],reference_method=previous['reference_method'],
                        comparison=compare_coordinate(gradients[i,j],previous['case']))
                if 'unconfirmed_case' in replacement:
                    evidence.update(native_confirmation=replacement['confirmation'],
                        initial_three_probe_comparison=compare_coordinate(gradients[i,j],replacement['unconfirmed_case']))
                report['refinements_applied'].append(evidence)
                reference=replacement['case']
            result=dict(index=j,name=str(names[j]),**compare_coordinate(gradients[i,j],reference))
            results.append(result);counts[result['status']]+=1
        certified=[r for r in results if 'max_relative_error' in r]
        report['points'].append(dict(index=i,status_counts=dict(Counter(r['status'] for r in results)),
            native_reference_report=point.get('reference_report',str(args.native)),
            max_relative_error_on_certified_references=max((r['max_relative_error'] for r in certified),default=None),
            unresolved_or_failed=[r for r in results if r['status'] not in ('passed','fixed_bounds')]))
    passed=report['nll_passed'] and shards_complete and len(report['points'])==26 and set(counts)<={'passed','fixed_bounds'}
    report.update(status='passed' if passed else 'incomplete_or_failed_native_gradient_gate',status_counts=dict(counts))
    save_json(args.output,report)
    print(report['status'],report['max_nll_absolute_error'],dict(counts),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
