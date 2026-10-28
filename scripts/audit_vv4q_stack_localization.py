"""Compare fixed Asimov channel/PDF exports and independently reconstruct templates."""
import json
from pathlib import Path
import numpy as np
from tensor_benchmark_common import sha,save_json


def load(p):return json.loads(Path(p).read_text())


def align(values,source_coordinates,target_coordinates):
    si=np.lexsort(source_coordinates.T[::-1]);ti=np.lexsort(target_coordinates.T[::-1])
    np.testing.assert_array_equal(source_coordinates[si],target_coordinates[ti])
    result=np.empty_like(values);result[ti]=values[si];return result


def main():
    paths={v:Path(f'reports/asymptotic_vv4q_stack_cached_{v}.json') for v in ('v10','v11')}
    reports={v:load(p) for v,p in paths.items()}
    for field in ('workspace_sha256','data_sha256','points_sha256','runtime_flags'):
        assert reports['v10'][field]==reports['v11'][field]
    for j in reports.values():assert j['status']=='completed' and len(j['cases'])==26
    arrays={v:np.load(p.with_suffix('.npz')) for v,p in paths.items()}
    for v in paths:assert sha(paths[v].with_suffix('.npz'))==reports[v]['arrays_sha256']
    result=dict(status='auditing',script_sha256=sha(__file__),
        reports={str(p):sha(p) for p in paths.values()},cases=[],pdf_cases=[],backend_checks={},histogram_checks={},
        scope='Identical workspace, fixed Asimov rows and parameter vectors; no fits. Process NLL substitution effects are not additive physical NLL terms.')
    result['cached_vs_scalar_max_absolute_error']={}
    for v in reports:
        scalar=np.load(f'reports/asymptotic_vv4q_stack_localization_{v}.npz')
        scalar_error=max(float(np.max(abs(scalar[k]-arrays[v][k]))) for k in scalar.files)
        assert scalar_error<1e-12
        result['cached_vs_scalar_max_absolute_error'][v]=scalar_error
    result['previous_shared_nll_reproduction']={}
    for v in reports:
        old=load(f'reports/asymptotic_vv4q_noVH_MX4000_{v}_shared_nll.json')
        err=max(abs(a['total']-b['nll']) for a,b in zip(reports[v]['cases'],old['cases']))
        assert err<1e-8
        result['previous_shared_nll_reproduction'][v]=err
    for a,b in zip(reports['v10']['cases'],reports['v11']['cases']):
        assert a['index']==b['index']
        channels={ch:b['channels'][ch]['nll']-a['channels'][ch]['nll'] for ch in a['channels']}
        delta=b['total']-a['total'];constraint=b['constraints']-a['constraints']
        assert abs(sum(channels.values())+constraint-delta)<1e-8
        result['cases'].append(dict(index=a['index'],delta_nll=delta,channels=channels,constraints=constraint))
        if a['index'] not in (0,1,23):continue
        for ch in channels:
            meta=reports['v10']['channels'][ch]
            names=[p['name'] for p in meta['pdfs'][:11]]
            w=arrays['v10'][ch+'__weights'];np.testing.assert_array_equal(w,arrays['v11'][ch+'__weights'])
            coeff=np.array(a['channels'][ch]['coefficients'])
            np.testing.assert_allclose(coeff,b['channels'][ch]['coefficients'],rtol=0,atol=1e-10)
            densities={v:np.stack([arrays[v][f"point{a['index']}__{ch}__{n}"] for n in names]) for v in arrays}
            mix={v:coeff@densities[v] for v in arrays}
            base_nll=-w@np.log(mix['v10']);target_nll=-w@np.log(mix['v11'])
            assert abs(target_nll-base_nll-channels[ch])<1e-6
            pdfs=[];sequential=mix['v10'].copy();prev=base_nll
            for i,n in enumerate(names):
                diff=densities['v11'][i]-densities['v10'][i]
                changed=coeff[i]*diff
                substitution=float(-w@np.log(mix['v10']+changed)-base_nll)
                sequential+=changed;current=-w@np.log(sequential)
                pdfs.append(dict(name=n,class_name=meta['pdfs'][i]['class_name'],
                    maximum_density_difference=float(max(abs(diff))),
                    maximum_relative_density_difference=float(max(abs(diff)/np.maximum(abs(densities['v10'][i]),1e-300))),
                    one_pdf_substitution_delta_nll=substitution,
                    sequential_substitution_delta_nll=float(current-prev)))
                prev=current
            result['pdf_cases'].append(dict(index=a['index'],channel=ch,delta_nll=channels[ch],pdfs=pdfs,
                reconstruction_error=float(target_nll-base_nll-channels[ch]),
                scope='Single-PDF substitutions start from the v10 mixture. Sequential substitutions follow the stored process order and telescope to the full channel difference.'))
    ref={c['index']:c for c in reports['v11']['cases']}
    for backend in ('legacy','cpu'):
        p=Path(f'reports/asymptotic_vv4q_stack_{backend}_v11.json');j=load(p)
        assert j['status']=='completed'
        differences=[(r['channels'][ch]['nll']-j['cases'][0]['channels'][ch]['nll'])-
            (ref[r['index']]['channels'][ch]['nll']-ref[0]['channels'][ch]['nll'])
            for r in j['cases'] for ch in r['channels']]
        result['backend_checks'][backend]=dict(report_sha256=sha(p),indices=[r['index'] for r in j['cases']],
            maximum_offset_subtracted_channel_error=float(max(abs(np.array(differences)))))
        assert max(abs(np.array(differences)))<1e-6
    for v in ('v10','v11'):
        hpath=Path(f'reports/asymptotic_vv4q_histogram_checked_{v}.json');h=load(hpath);ha=np.load(hpath.with_suffix('.npz'))
        assert h['status']=='completed' and h['workspace_sha256']==reports[v]['workspace_sha256']
        checks={}
        for ch,meta in h['channels'].items():
            assert meta['smooth_algo']==-1 and meta['smooth_region']==1 and not meta['conditional']
            stored=ha[ch+'__template5'];vol=stored[:,1];measured=ha[ch+'__stored'][:,-1]
            density=stored[:,0]/np.sum(stored[:,0])/vol
            mass=stored[:,0]/np.sum(stored[:,0]*vol)
            checks[ch]=dict(density_max_abs=float(max(abs(density-measured))),
                mass_as_density_max_abs=float(max(abs(mass-measured))),
                stored_density_l1_probability=float(np.sum(abs(density-measured)*vol)),
                bin_volume_min=float(vol.min()),bin_volume_max=float(vol.max()))
            coords=np.column_stack([arrays[v][ch+'__coord__'+n] for n in meta['observables']])
            for case in (0,1,23):
                point=load('build/vv4q_version_points.json')
                theta=dict(zip(point['parameter_names'],point['cases'][case]['point']))
                ts=np.stack([ha[f'{ch}__template{i}'][:,0] for i in range(11)])
                # Emulate native TH3F storage only at the ROOT histogram boundary.
                # All analysis and model arithmetic remains float64.
                if v=='v10':
                    avg=np.prod([len(np.unique(stored[:,i])) for i in (2,3,4)])
                    mean_volume=float(vol.sum()/avg)
                    ts=(ts/ts.sum(axis=1)[:,None]/vol*mean_volume).astype(np.float32).astype(np.float64)
                else:ts=(ts/ts.sum(axis=1)[:,None]).astype(np.float32).astype(np.float64)
                ts/=np.sum(ts*vol,axis=1)[:,None]
                log0=np.log(ts[0]);pred=log0.copy()
                for i,name in enumerate(meta['coefficients']):
                    x=theta[name];smooth=np.sign(x) if abs(x)>=1 else .125*x*(x*x*(3*x*x-10)+15)
                    hi=np.log(ts[2*i+1])-log0;lo=np.log(ts[2*i+2])-log0
                    pred+=.5*x*((hi-lo)+smooth*(hi+lo))
                pred=np.exp(pred);pred/=np.sum(pred*vol)
                predicted=align(pred,stored[:,2:5],coords)
                observed=arrays[v][f"point{case}__{ch}__shapeBkg_nonRes_{ch}"]
                checks[ch][f'point{case}_numpy_max_abs']=float(max(abs(predicted-observed)))
                checks[ch][f'point{case}_numpy_max_relative']=float(max(abs(predicted-observed)/np.maximum(abs(observed),1e-300)))
        result['histogram_checks'][v]=dict(report_sha256=sha(hpath),channels=checks)
    left=np.load('reports/asymptotic_vv4q_histogram_checked_v10.npz')
    right=np.load('reports/asymptotic_vv4q_histogram_checked_v11.npz')
    result['stored_template_contents_identical']=all(np.array_equal(left[f'{ch}__template{i}'][:,:5],right[f'{ch}__template{i}'][:,:5])
        for ch in reports['v10']['channels'] for i in range(11))
    assert result['stored_template_contents_identical']
    one={v:load(f'reports/asymptotic_vv4q_1d_{v}.json') for v in reports}
    assert all(j['status']=='completed' and len(j['pdfs'])==40 for j in one.values())
    assert all(one['v10']['pdfs'][n]['stored_values_sha256']==p['stored_values_sha256'] for n,p in one['v11']['pdfs'].items())
    result['one_dimensional_checks']={}
    for v,convention in [('v10','density'),('v11','mass')]:
        err=max(p['predictions'][convention]['max_relative'] for p in one[v]['pdfs'].values())
        assert err<1e-12
        result['one_dimensional_checks'][v]=dict(pdfs=40,matching_numpy_convention=convention,
            max_relative_error=err,stored_bin_contents_identical=True,
            report_sha256=sha(f'reports/asymptotic_vv4q_1d_{v}.json'))
    generated=np.load('build/vv4q_asimov_generation_rows.npz')
    export=load('reports/asymptotic_vv4q_asimov_row_export.json')
    assert sha(export['arrays'])==export['arrays_sha256']
    result['generation_checks']={}
    for v in ('v10','v11'):
        ha=np.load(f'reports/asymptotic_vv4q_histogram_checked_{v}.npz');checks={}
        for ch,meta in reports[v]['channels'].items():
            obs=['MJ1','MJ2','MJJ'];coords=np.column_stack([arrays[v][ch+'__coord__'+n] for n in obs])
            stored=ha[ch+'__stored'];vol=align(stored[:,1],stored[:,2:5],coords)
            coeff=np.array(reports[v]['cases'][0]['channels'][ch]['coefficients'])
            assert np.all(coeff[:2]==0), 'Background-only point required'
            mix=coeff@np.stack([arrays[v]['point0__'+ch+'__'+p['name']] for p in meta['pdfs'][:11]])
            dc=np.column_stack([generated[v+'__'+ch+'__'+n] for n in obs])
            target=align(generated[v+'__'+ch+'__weights'],dc,coords)
            # Extended createHistogram: ROOT 6.30 disables density scaling;
            # ROOT 6.40 keeps it, then ToyMCSamplerOpt multiplies volume again.
            h=(mix*(vol if v=='v11' else 1.)).astype(np.float32).astype(np.float64)
            predicted=(h*coeff.sum()/np.sum(h*vol)).astype(np.float32).astype(np.float64)*vol
            density_asimov=mix*vol;density_asimov*=coeff.sum()/density_asimov.sum()
            checks[ch]=dict(native_recipe_max_weight_error=float(max(abs(predicted-target))),
                native_recipe_l1_weight_error=float(sum(abs(predicted-target))),
                normalized_density_asimov_l1_weight_difference=float(sum(abs(density_asimov-target))))
        result['generation_checks'][v]=checks
    result.update(status='completed',source_conventions={
        'root_630_source':'https://raw.githubusercontent.com/root-project/root/v6-30-06/roofit/roofitcore/src/RooAbsReal.cxx',
        'root_current_source':'https://root.cern.ch/doc/master/RooAbsReal_8cxx_source.html',
        'note':'6.30 source multiplies by average bin volume; current source multiplies each bin volume. Measured 6.30.07 and 6.40.00 histograms independently verify the change. Combine VerticalInterpHistPdf and ToyMCSamplerOpt then normalize with bin widths again.'})
    save_json(Path('reports/asymptotic_vv4q_stack_localization_audit.json'),result)
    print('nominal',result['cases'][0]);print('largest',max(result['cases'],key=lambda c:abs(c['delta_nll'])))


if __name__=='__main__':main()
