"""Density-preserving Asimov rows on explicitly checked native histogram grids."""
from functools import lru_cache

import numpy as np


def variable_bin_channels(workspace):
    """Detect the variable-grid channels for which native histogram generation differs."""
    from ..root_io import load_root
    root = load_root()
    mc = workspace.obj('ModelConfig'); pdf = mc.GetPdf()
    if pdf.ClassName() not in ('RooSimultaneous', 'RooSimultaneousOpt'):
        raise NotImplementedError('Expected a simultaneous model for variable-bin detection')
    category = pdf.indexCat().Clone(); affected = {}
    for i in range(category.numTypes()):
        category.setBin(i); label = str(category.getLabel())
        channel = pdf.getPdf(label)
        obs = channel.getObservables(mc.GetObservables())
        edges = {}
        for v in obs:
            if v.ClassName() != 'RooRealVar': continue
            b = v.getBinning()
            e = np.array([b.binLow(j) for j in range(b.numBins())]+[b.highBound()])
            edges[str(v.GetName())] = e
        if any(not np.allclose(np.diff(e), np.diff(e)[0], rtol=1e-12, atol=0.) for e in edges.values()):
            if not 1 <= len(edges) <= 3 or len(edges) != len(obs):
                raise NotImplementedError('Density Asimov generation requires 1-3 real histogram axes')
            affected[label] = edges
    return affected


@lru_cache(maxsize=1)
def writer():
    from ..root_io import load_root
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "TH3.h"
      #include "RooDataSet.h"
      namespace combine_jax_io {
      std::vector<double> densityAsimovWeights(RooArgList const& axes,
          std::vector<std::vector<double>> const& coords, std::vector<double> const& density, double yield) {
        int dim=axes.size();
        auto& x=dynamic_cast<RooRealVar&>(*axes.at(0));
        RooCmdArg y=dim>1 ? RooFit::YVar(dynamic_cast<RooRealVar&>(*axes.at(1))) : RooCmdArg::none();
        RooCmdArg z=dim>2 ? RooFit::ZVar(dynamic_cast<RooRealVar&>(*axes.at(2))) : RooCmdArg::none();
        std::unique_ptr<TH1> h(x.createHistogram("combine_jax_asimov_density",y,z)); h->SetDirectory(nullptr);
        std::vector<int> bins; std::vector<double> volumes;
        for(size_t i=0;i<density.size();++i) {
          int bx=h->GetXaxis()->FindBin(coords[0][i]);
          int by=dim>1 ? h->GetYaxis()->FindBin(coords[1][i]) : 1;
          int bz=dim>2 ? h->GetZaxis()->FindBin(coords[2][i]) : 1;
          int bin=h->GetBin(bx,by,bz); bins.push_back(bin);
          double volume=h->GetXaxis()->GetBinWidth(bx);
          if(dim>1) volume*=h->GetYaxis()->GetBinWidth(by);
          if(dim>2) volume*=h->GetZaxis()->GetBinWidth(bz);
          volumes.push_back(volume);
          // Native ROOT 6.30 extended histogram stores density * expectedEvents
          // in TH*F, then ToyMCSamplerOpt normalizes once and multiplies volume.
          h->SetBinContent(bin,density[i]*yield);
        }
        if(!(h->Integral("width")>0.)) throw std::invalid_argument("Empty Asimov density");
        h->Scale(yield/h->Integral("width"));
        std::vector<double> result;
        for(size_t i=0;i<bins.size();++i) result.push_back(h->GetBinContent(bins[i])*volumes[i]);
        return result;
      }
      RooDataSet* replaceAsimovWeights(RooDataSet const& data, std::string const& category,
          std::vector<std::string> const& labels, std::vector<std::vector<double>> const& weights) {
        auto* out=dynamic_cast<RooDataSet*>(data.emptyClone("combine_jax_density_asimov"));
        std::vector<size_t> cursor(labels.size(),0);
        for(int i=0;i<data.numEntries();++i) {
          auto* row=data.get(i); double weight=data.weight();
          auto const& cat=dynamic_cast<RooAbsCategory const&>(*row->find(category.c_str()));
          for(size_t j=0;j<labels.size();++j) if(labels[j]==cat.getCurrentLabel()) {
            weight=weights[j].at(cursor[j]++); break;
          }
          out->add(*row,weight);
        }
        for(size_t j=0;j<labels.size();++j) if(cursor[j]!=weights[j].size()) throw std::invalid_argument("Asimov row count mismatch");
        return out;
      }
      }
    '''):
        raise RuntimeError('Cannot compile Asimov row writer')
    return root.combine_jax_io


def correct_asimov(workspace, generated, affected):
    """Use corrected exported densities and v10 histogram-boundary arithmetic.

    Does not fit, change global observables, or mutate the workspace dataset.
    Refuses non-grid/pseudo-Asimov rows instead of approximating their weights.
    """
    import jax
    from ..likelihood import from_workspace
    from ..registry import Evaluation
    from ..translators import default_registry
    from ..root_io import load_root
    root = load_root(); helper = writer()
    likelihood = from_workspace(workspace, data=generated,
        registry=default_registry(template_convention='v10_density'))
    names = [p.name for p in likelihood.parameters]
    weights = []; labels = []; details = {}
    for channel in likelihood.channels:
        if channel.name not in affected: continue
        edges = affected[channel.name]
        coordinates = channel.data.coordinates
        obs_names = sorted(edges)
        shape = tuple(len(edges[n])-1 for n in obs_names)
        if len(channel.data.weights) != np.prod(shape):
            raise NotImplementedError(f'{channel.name}: Asimov rows are not a complete histogram grid')
        indices = []
        for n in obs_names:
            e = edges[n]; index = np.searchsorted(e,coordinates[n],side='right')-1
            if (np.any(index<0) or np.any(index>=len(e)-1)
                    or not np.allclose(coordinates[n],(e[index]+e[index+1])/2,rtol=1e-12,atol=1e-12)):
                raise NotImplementedError(f'{channel.name}: Asimov coordinates are not bin centers')
            indices.append(index)
        if len(np.unique(np.ravel_multi_index(indices,shape))) != np.prod(shape):
            raise ValueError('Duplicate Asimov grid bins')
        def density(theta):
            return Evaluation(likelihood.nodes,likelihood.registry,names,theta,coordinates).density(channel.pdf,channel.observables)
        values = np.asarray(jax.jit(density)(likelihood.initial))
        native_pdf = workspace.pdf(channel.pdf)
        axes = [workspace.var(n) for n in obs_names]
        expected = float(native_pdf.expectedEvents(root.RooArgSet(*axes)))
        result = np.asarray(helper.densityAsimovWeights(root.RooArgList(*axes),
            [coordinates[n].tolist() for n in obs_names],values.tolist(),expected)).copy()
        if np.any(result<0) or not np.all(np.isfinite(result)):
            raise ValueError('Invalid corrected Asimov weights')
        labels.append(channel.name); weights.append(result.tolist())
        details[channel.name] = dict(rows=len(result),sum_weights=float(result.sum()),expected_events=expected,
            maximum_weight_change=float(np.max(abs(result-channel.data.weights))),
            l1_weight_change=float(np.sum(abs(result-channel.data.weights))))
    category = workspace.obj('ModelConfig').GetPdf().indexCat().GetName()
    result = helper.replaceAsimovWeights(generated,category,labels,weights)
    root.SetOwnership(result,True)
    return result, dict(convention='v10_density',channels=details,
        arithmetic='Float64 model densities; native ROOT 6.30 TH*F boundary rounding, one volume factor. No reduced-precision model arrays.')
