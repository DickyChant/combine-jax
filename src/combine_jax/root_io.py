"""The checked, wheel-aware ROOT boundary; no dependencies from JAX kernels."""

from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path


# Likelihood evaluator defaults from Combine v11 bin/combine.cpp, lines 317-333.
REFERENCE_FLAGS = {
    name: 1 for name in (
        "ADDNLL_RECURSIVE", "ADDNLL_GAUSSNLL", "ADDNLL_HISTNLL", "ADDNLL_CBNLL",
        "ADDNLL_ROOREALSUM_FACTOR", "ADDNLL_ROOREALSUM_NONORM", "ADDNLL_ROOREALSUM_BASICINT",
        "ADDNLL_ROOREALSUM_KEEPZEROS", "ADDNLL_PRODNLL", "ADDNLL_HFNLL",
        "ADDNLL_HISTFUNCNLL", "ADDNLL_ROOREALSUM_CHEAPPROD",
    )
}


@lru_cache(maxsize=1)
def load_root():
    from cms_combine.cli import runtime_environment
    import ROOT

    env = runtime_environment()
    # Headers must be available before Cling loads the dictionary.
    for directory in env["ROOT_INCLUDE_PATH"].split(":"):
        ROOT.gInterpreter.AddIncludePath(directory)
    libraries = [Path(p) / "libHiggsAnalysisCombinedLimit.so"
                 for p in env["LD_LIBRARY_PATH"].split(":")]
    library = next((p for p in libraries if p.is_file()), None)
    if library is None or ROOT.gSystem.Load(str(library)) < 0:
        raise RuntimeError("Cannot load the installed Combine dictionary; no workspace opened")
    ROOT.gROOT.SetBatch(True)
    return ROOT


@contextmanager
def open_workspace(path, name="w"):
    root = load_root()
    handle = root.TFile.Open(str(path), "READ")
    if not handle or handle.IsZombie():
        raise OSError(f"Cannot open ROOT file: {path}")
    try:
        workspace = handle.Get(name)
        if not workspace or not workspace.InheritsFrom("RooWorkspace"):
            raise ValueError(f"{path}: no RooWorkspace named {name!r}")
        yield workspace
    finally:
        handle.Close()


@lru_cache(maxsize=1)
def load_caching_nll():
    root = load_root()
    if not root.gInterpreter.Declare(
        '#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"'
    ):
        raise RuntimeError("Cannot load Combine CachingNLL declarations")
    for flag, value in REFERENCE_FLAGS.items():
        root.runtimedef.set(flag, value)
    return root


@lru_cache(maxsize=1)
def member_reader():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "TBufferJSON.h"
      namespace combine_jax_io {
      TString storedMember(RooAbsArg const& node, char const* name) {
        struct FormatGuard {
          std::string saved{TBufferJSON::GetDoubleFormat()};
          FormatGuard() { TBufferJSON::SetDoubleFormat("%.17g"); }
          ~FormatGuard() { TBufferJSON::SetDoubleFormat(saved.c_str()); }
        } guard;
        return TBufferJSON::ConvertToJSON(&node, 3, name);
      }
      }
    '''):
        raise RuntimeError("Cannot compile stored-member serialization helper")
    return root.combine_jax_io.storedMember


def stored_member(node, name):
    """Read dictionary-persisted state at round-trip precision, including private members."""
    return json.loads(str(member_reader()(node, name)))
