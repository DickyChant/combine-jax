"""Bind actual RooFit rows without assuming they are histogram bin centres."""

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

import numpy as np

from .graph import UnsupportedModelError
from .root_io import load_root


@dataclass(frozen=True)
class DataBinding:
    coordinates: object
    weights: np.ndarray
    storage_class: str


@lru_cache(maxsize=1)
def data_reader():
    root=load_root()
    if not root.gInterpreter.Declare(r'''
      #include "RooAbsData.h"
      #include "RooAbsCategory.h"
      #include "RooAbsReal.h"
      #include "RooArgSet.h"
      #include <unordered_map>
      #include <stdexcept>
      namespace combine_jax_io {
      using Columns = std::vector<std::vector<double>>;
      std::vector<Columns> readRows(RooAbsData const& data, std::string const& categoryName,
                                    std::vector<std::string> const& labels,
                                    std::vector<std::vector<std::string>> const& names) {
        auto const* schema=data.get();
        RooAbsCategory const* category=nullptr;
        if (!categoryName.empty()) {
          category=dynamic_cast<RooAbsCategory const*>(schema->find(categoryName.c_str()));
          if (!category) throw std::invalid_argument("Missing category "+categoryName);
        }
        std::unordered_map<int,size_t> labelMap;
        std::vector<std::vector<RooAbsReal const*>> variables(names.size());
        std::vector<Columns> result(names.size());
        for (size_t channel=0; channel<names.size(); ++channel) {
          if (category) {
            if (!category->hasLabel(labels[channel])) throw std::invalid_argument("Unknown category label "+labels[channel]);
            if (!labelMap.emplace(category->lookupIndex(labels[channel]),channel).second)
              throw std::invalid_argument("Duplicate category label "+labels[channel]);
          }
          result[channel].resize(names[channel].size()+1);
          for (auto const& name:names[channel]) {
            auto* variable=dynamic_cast<RooAbsReal const*>(schema->find(name.c_str()));
            if (!variable) throw std::invalid_argument("Missing real observable "+name);
            variables[channel].push_back(variable);
          }
        }
        for (int row=0; row<data.numEntries(); ++row) {
          data.get(row);
          size_t channel=0;
          if (category) {
            auto found=labelMap.find(category->getCurrentIndex());
            if (found==labelMap.end()) continue;
            channel=found->second;
          }
          auto& columns=result[channel];
          for (size_t i=0;i<variables[channel].size();++i) columns[i].push_back(variables[channel][i]->getVal());
          columns.back().push_back(data.weight());
        }
        return result;
      }
      }
    '''):
        raise RuntimeError("Cannot compile bulk data reader")
    return root.combine_jax_io.readRows


def checked_storage(dataset):
    if dataset.ClassName() not in ("RooDataSet", "RooDataHist"):
        raise UnsupportedModelError(f"Unsupported data storage {dataset.ClassName()}")


def binding_from_columns(columns, observables, storage_class):
    # Own the exported buffers independently of ROOT and the temporary vectors.
    coordinates={name:np.array(column,dtype=np.float64,copy=True) for name,column in zip(observables,columns[:-1],strict=True)}
    weights=np.array(columns[-1],dtype=np.float64,copy=True)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise UnsupportedModelError("Negative or nonfinite event weights are not yet supported")
    if any(not np.all(np.isfinite(column)) for column in coordinates.values()):
        raise ValueError("Nonfinite observable coordinates")
    for array in (*coordinates.values(), weights):
        array.flags.writeable = False
    return DataBinding(MappingProxyType(coordinates), weights, storage_class)


def bind_data(dataset, observables):
    checked_storage(dataset)
    observables=tuple(sorted(observables))
    columns=data_reader()(dataset,"",[],[list(observables)])
    return binding_from_columns(list(columns[0]),observables,str(dataset.ClassName()))


def bind_channels(dataset, category, channels):
    """Export requested channel columns and weights in one dataset traversal."""
    checked_storage(dataset)
    labels=list(channels)
    observables=[tuple(sorted(channels[label])) for label in labels]
    columns=data_reader()(dataset,category,labels,[list(names) for names in observables])
    return {label:binding_from_columns(list(column),names,str(dataset.ClassName()))
            for label,column,names in zip(labels,columns,observables,strict=True)}
