"""Native template-byte comparison and fixed-step clipping probes at MX4000."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from tensor_benchmark_common import Resources, provenance, save_json, sha


CPP = r'''
#include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
#include "RooDataHist.h"
#include "TBufferJSON.h"
namespace mass_clipping_audit {
TString geometry(FastHisto2D const& h, char const* member) {
  std::string saved(TBufferJSON::GetDoubleFormat());
  TBufferJSON::SetDoubleFormat("%.17g");
  auto result=TBufferJSON::ConvertToJSON(&h,TClass::GetClass<FastHisto2D>(),3,member);
  TBufferJSON::SetDoubleFormat(saved.c_str()); return result;
}
std::vector<double> values(FastTemplate const& h) {
  std::vector<double> out; for(unsigned i=0;i<h.fullsize();++i) out.push_back(h[i]); return out;
}
class View2D : public FastVerticalInterpHistPdf2D2 {
public:
  View2D(FastVerticalInterpHistPdf2D2 const& p) : FastVerticalInterpHistPdf2D2(p,"mass_audit_view") {}
  std::vector<double> raw() const {
    evaluate();
    FastHisto2D h(_cacheNominal);
    if(_smoothAlgo<0) throw std::runtime_error("Only additive clipping audited");
    for(int i=0;i<_coefList.getSize();++i) {
      double x=dynamic_cast<RooAbsReal const*>(_coefList.at(i))->getVal();
      h.Meld(_morphs[i].diff,_morphs[i].sum,0.5*x,smoothStepFunc(x));
    }
    return values(h);
  }
  std::vector<double> normalized() const { evaluate(); return values(_cache); }
};
std::vector<double> data(RooDataHist const& h) {
  std::vector<double> out;
  for(int i=0;i<h.numEntries();++i) {
    auto row=h.get(i);
    for(auto arg:*row) out.push_back(dynamic_cast<RooAbsReal const*>(arg)->getVal());
    out.push_back(h.weight());
  }
  return out;
}
}
'''


def digest_values(values):
    array = np.asarray(values, dtype='<f8')
    return dict(size=int(array.size), bytes=int(array.nbytes), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def template_inventory(workspace, root):
    parents = [p for p in workspace.allPdfs() if p.GetName().startswith('shapeBkg_')
               and ('WW_lvqq' in p.GetName() or 'bbww_boosted' in p.GetName())]
    nodes = {}
    for parent in parents:
        components = parent.getComponents()
        for node in components:
            nodes[node.GetName()] = workspace.arg(node.GetName())
    result = {}
    for name, node in sorted(nodes.items()):
        cls = node.ClassName()
        if cls not in ('RooHistPdf', 'FastVerticalInterpHistPdf', 'FastVerticalInterpHistPdf2D2'):
            continue
        row = dict(class_name=cls, buffers={})
        if cls == 'RooHistPdf':
            hist = node.dataHist()
            row['buffers']['coordinates_and_weights'] = digest_values(root.mass_clipping_audit.data(hist))
            row['axes'] = []
            for axis in hist.get():
                bins = axis.getBinning()
                row['axes'].append(axis.GetName())
                row['buffers']['edges_'+axis.GetName()] = digest_values(
                    [bins.binLow(i) for i in range(bins.numBins())]+[bins.binHigh(bins.numBins()-1)])
        else:
            node.evaluate()
            nominal = node.cacheNominal()
            row.update(coefficients=[c.GetName() for c in node.coefList()],
                       smooth_algorithm=node.smoothAlgo(), smooth_region=node.smoothRegion(),
                       active_bins=int(nominal.size()))
            row['buffers']['nominal'] = digest_values(root.mass_clipping_audit.values(nominal))
            for i, morph in enumerate(node.morphs()):
                for key in ('sum', 'diff'):
                    row['buffers'][f'{key}_{i}'] = digest_values(root.mass_clipping_audit.values(getattr(morph, key)))
            if cls == 'FastVerticalInterpHistPdf2D2':
                row['conditional'] = bool(node.conditional())
                for member in ('binEdgesX_', 'binEdgesY_', 'binWidths_'):
                    row['buffers'][member] = digest_values(json.loads(str(root.mass_clipping_audit.geometry(nominal, member))))
            else:
                row['buffers']['edges'] = digest_values([nominal.GetEdge(i) for i in range(nominal.fullsize()+1)])
                row['buffers']['widths'] = digest_values([nominal.GetWidth(i) for i in range(nominal.fullsize())])
        result[name] = row
    unsupported = {name: node.ClassName() for name, node in nodes.items()
                   if 'Hist' in node.ClassName() and name not in result}
    assert not unsupported, unsupported
    return dict(parent_names=sorted(p.GetName() for p in parents), classes=dict(Counter(n.ClassName() for n in nodes.values())),
                templates=result, unsupported=unsupported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    assert root.gInterpreter.Declare(CPP)
    prior_path = Path('reports/gpu_tensor_reduced_fit_clipping_geometry_audit.json')
    prior = json.loads(prior_path.read_text())
    scales = {}
    for path, digest in prior['native_probes'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['points']:
            scales[row['name']] = row['scale']
    paths = {2000: Path('inputs/workspace_MX2000.root'), 4000: Path('inputs/snapshot_MX4000.root')}
    report = dict(provenance(), status='running', script_sha256=sha(__file__), combine_version='11.0.0', root_version='6.40.00',
        workspaces={str(m): dict(path=str(p), sha256=sha(p)) for m,p in paths.items()},
        prior_audit=str(prior_path), prior_audit_sha256=sha(prior_path),
        runtime_flags=REFERENCE_FLAGS, floor=1e-9, step_fractions=[10.**(-k) for k in range(2,9)], inventories={}, probes=[],
        byte_comparison_scope='Exact little-endian float64 payload bytes: RooDataHist coordinates/weights/edges and native morpher nominal/sum/diff/edges/widths, with matching coefficient order and interpolation metadata. Not whole ROOT file serialization. Parametric RooDoubleCB functions are listed separately, not called templates.',
        point_scope='MX4000 workspace MultiDimFit saved snapshot, data_obs. Saved snapshot is not a new fit or convergence certificate; original fit dataset identity is not independently recoverable from the empty toys directory.')
    monitor = Resources(False, args.output)
    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)
    save()
    try:
        for mass, path in paths.items():
            monitor.stage = 'templates_'+str(mass)
            with open_workspace(path) as workspace:
                report['inventories'][str(mass)] = template_inventory(workspace, root)
                save()
        left, right = (report['inventories'][str(m)] for m in (2000, 4000))
        names = sorted(set(left['templates']) | set(right['templates']))
        differences = [n for n in names if left['templates'].get(n) != right['templates'].get(n)]
        report['template_comparison'] = dict(compared=len(names), byte_identical=len(names)-len(differences),
            differing= differences, same_parent_names=left['parent_names']==right['parent_names'])
        monitor.stage = 'saved_snapshot_native_probes'; save()
        with open_workspace(paths[4000]) as workspace:
            snapshot = workspace.getSnapshot('MultiDimFit')
            assert snapshot
            workspace.loadSnapshot('MultiDimFit')
            assert workspace.var('r_hhvv').getVal() == snapshot.find('r_hhvv').getVal()
            config = workspace.obj('ModelConfig'); data = workspace.data('data_obs')
            report['snapshot'] = dict(name='MultiDimFit', variables=len(snapshot), MX=workspace.var('MX').getVal(),
                r_hhvv=workspace.var('r_hhvv').getVal(), dataset=data.GetName(), entries=data.numEntries(), sum_weights=data.sumEntries())
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), data, config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt': nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True); nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
            def value():
                root.RooAbsReal.clearEvalErrorLog(); v=float(nll.getVal())
                assert np.isfinite(v) and root.RooAbsReal.numEvalErrors()==0
                return v
            canonical=value(); report['canonical_nll']=canonical
            nll.setZeroPoint(); baseline=value(); assert abs(baseline)<1e-6
            for surface in prior['surfaces']:
                pdf=workspace.pdf(surface['histogram']); assert pdf.ClassName()=='FastVerticalInterpHistPdf2D2'
                view=root.mass_clipping_audit.View2D(pdf); k=surface['flat_bin']
                variable=workspace.var(surface['dominant_parameter']); origin=variable.getVal()
                # Match the earlier seven-step, scale-based native probe protocol.
                scale=scales[variable.GetName()]
                row=dict(histogram=surface['histogram'], flat_bin=k, parameter=variable.GetName(), coordinate=origin,
                    scale=scale, raw_bin=float(view.raw()[k]), normalized_bin=float(view.normalized()[k]), probes=[])
                row['distance_to_floor']=row['raw_bin']-1e-9
                row['clipped_at_saved_point']=row['raw_bin']<1e-9
                report['probes'].append(row)
                for fraction in report['step_fractions']:
                    h=fraction*scale
                    assert variable.getMin()<=origin-2*h<origin+2*h<=variable.getMax()
                    deltas={}; bins={}
                    for multiple in (-2,-1,1,2):
                        variable.setVal(origin+multiple*h); deltas[multiple]=value()-baseline; bins[multiple]=float(view.raw()[k])
                    variable.setVal(origin); restored=value()-baseline; assert abs(restored)<1e-6
                    backward=(-4*deltas[-1]+deltas[-2])/(2*h); forward=(4*deltas[1]-deltas[2])/(2*h)
                    row['probes'].append(dict(step=h, deltas=deltas, raw_bins=bins, central=(deltas[1]-deltas[-1])/(2*h),
                        backward=backward, forward=forward, signs_bracket_zero=backward<0<forward,
                        crosses_floor=min(bins.values())<1e-9<max(bins.values()), restoration_absolute_error=abs(restored)))
                    save()
                print(row['histogram'], k, row['raw_bin'], row['clipped_at_saved_point'], flush=True)
            nll.clearZeroPoint(); nll.clearConstantZeroPoint()
            report['canonical_restoration_absolute_error']=abs(value()-canonical)
            assert report['canonical_restoration_absolute_error']<1e-6
            nll.setAnalyticBarlowBeeston(False)
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='native_snapshot_probes_and_template_comparison_complete', jax_imported=False,
            clipped_selected_bins=sum(p['clipped_at_saved_point'] for p in report['probes']),
            selected_bins_with_floor_crossing=sum(any(s['crosses_floor'] for s in p['probes']) for p in report['probes']))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
