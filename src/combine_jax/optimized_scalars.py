"""Exact optimized Combine scalar expressions, including stored folded factors."""
from functools import lru_cache

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .root_io import load_root


HGG_CLASSES = tuple('CMSHggFormula' + name for name in ('A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'D2'))


@lru_cache(maxsize=1)
def optimized_scalar_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/CMSHggFormula.h"
      #include "HiggsAnalysis/CombinedLimit/interface/RooCheapProduct.h"
      #include <type_traits>
      namespace combine_jax_io {
      inline std::string scalarFieldName(RooRealProxy const& p) { return p.arg().GetName(); }
      inline std::string scalarFieldName(double) { return ""; }
      inline double scalarFieldLiteral(RooRealProxy const&) { return 0.; }
      inline double scalarFieldLiteral(double p) { return p; }
      template<class T> class HggAccess : public T {
        static constexpr bool isA = std::is_same_v<T,CMSHggFormulaA1> || std::is_same_v<T,CMSHggFormulaA2>;
        static constexpr bool isB = std::is_same_v<T,CMSHggFormulaB1> || std::is_same_v<T,CMSHggFormulaB2>;
        static constexpr bool isD = std::is_same_v<T,CMSHggFormulaD1> || std::is_same_v<T,CMSHggFormulaD2>;
      public:
        HggAccess(T const& p) : T(p,"combine_jax_hgg_access") {}
        std::vector<std::string> fields() const {
          if constexpr (isA) return {scalarFieldName(this->p0_),scalarFieldName(this->p1_),
                                     scalarFieldName(this->p2_),scalarFieldName(this->p3_)};
          else if constexpr (isB) return {scalarFieldName(this->p0_)};
          else if constexpr (isD) return {scalarFieldName(this->p0_),scalarFieldName(this->p1_)};
          else return {};
        }
        std::vector<double> literals() const {
          if constexpr (isA) return {scalarFieldLiteral(this->p0_),scalarFieldLiteral(this->p1_),
                                     scalarFieldLiteral(this->p2_),scalarFieldLiteral(this->p3_)};
          else if constexpr (isB) return {scalarFieldLiteral(this->p0_)};
          else if constexpr (isD) return {scalarFieldLiteral(this->p0_),scalarFieldLiteral(this->p1_)};
          else return {};
        }
        std::vector<std::string> terms() const {
          std::vector<std::string> names;
          if constexpr (!isD) for (auto* term : this->terms_) names.emplace_back(term->GetName());
          return names;
        }
        std::vector<double> coefficients() const {
          if constexpr (isD) return {};
          else return this->coeffs_;
        }
      };
      class CheapProductAccess : public RooCheapProduct {
      public:
        CheapProductAccess(RooCheapProduct const& p) : RooCheapProduct(p,"combine_jax_cheap_product") {}
        double foldedFactor() const { return offset_; }
      };
      }
    '''):
        raise RuntimeError('Cannot compile optimized scalar extraction helpers')
    return root.combine_jax_io


def extract_hgg(node):
    cls = str(node.ClassName())
    if cls not in HGG_CLASSES:
        raise UnsupportedModelError(f'Unsupported optimized Hgg formula {cls}')
    view = optimized_scalar_access().HggAccess[cls](node)
    fields = tuple(str(x) or None for x in view.fields())
    literals = tuple(float(x) for x in view.literals())
    terms = tuple(str(x) for x in view.terms())
    coefficients = tuple(float(x) for x in view.coefficients())
    if len(terms) != len(coefficients):
        raise UnsupportedModelError(f'{node.GetName()}: Hgg term/coefficient count mismatch')
    return tuple(x for x in fields if x) + terms, dict(
        kind=cls[-2], fields=fields, literals=literals, terms=terms, coefficients=coefficients)


def hgg_value(node, context):
    attrs = node.attrs
    fields = [context.value(name) if name else jnp.asarray(literal)
              for name, literal in zip(attrs['fields'], attrs['literals'], strict=True)]
    kind = attrs['kind']
    if kind == 'D':
        total = fields[0] + fields[1]
        # std::min(a,b) selects its first argument at equality. Do not smooth
        # the physical kink or use jnp.minimum's half-gradient convention.
        return jnp.where(total <= 1., total, 1.)
    result = 1. + fields[2] + fields[3] if kind == 'A' else jnp.asarray(1.)
    for coefficient, name in zip(attrs['coefficients'], attrs['terms'], strict=True):
        result = result + coefficient * context.value(name)
    if kind == 'A':
        return (fields[0] + fields[1]) * result
    if kind == 'B':
        # std::max(1.e-2,ret) keeps the constant first argument at equality.
        return fields[0] * jnp.where(result > .01, result, .01)
    return result


def extract_cheap_product(node):
    view = optimized_scalar_access().CheapProductAccess(node)
    return tuple(str(x.GetName()) for x in view.components()), {'folded_factor': float(view.foldedFactor())}


def cheap_product_value(node, context):
    result = jnp.asarray(node.attrs['folded_factor'])
    for name in node.inputs:
        result = result * context.value(name)
    return result
