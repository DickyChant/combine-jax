---
name: ad-port-and-validate
description: Port a numerical model from a reference implementation (RooFit/Combine, C++, Fortran, TF) into an autodiff framework (JAX/torch) and prove that values AND derivatives match the reference. Use when writing converters for many function classes, when an AD gradient disagrees with a finite-difference reference, or when deciding whether a mismatch is a port bug or reference noise.
---

# Port to autodiff and validate derivatives

Learned on combine-jax (HHVV, HIG-23-006 H+HH: 46 RooFit classes, ~200k nodes,
11.5k fit parameters). The LLM writes the converters; the reference decides what is right.

## Ground rules
- The reference implementation is the truth. Pin its version (e.g. Combine v10.3.3,
  ROOT 6.30/07) and record it next to every number.
- AD gives the exact derivative **of the code you wrote**. The work is proving that
  code equals the reference function; you rarely need to derive derivatives by hand.
- Never loosen a tolerance or change a finite-difference step to make a gate pass.
  Diagnose instead, and keep the failing result on record.
- float64 throughout. Every claimed agreement needs a test or a logged comparison.
- Never silently approximate an unsupported class: fail loudly with its name.

## 1. Census before code
List every class/op in the real model (with counts and parameter roles) before writing
converters. The census is the coverage checklist; an unknown class is an import error,
not a fallback.

## 2. One converter per class, each with its own fixture
For each class:
1. Read the reference **source** for the version you pin, not the docs: branch
   thresholds, floors, clipping, special substitutions (e.g. ErfExp's zero-slope
   −1e-7), signed tails, normalization conventions, edge-bin inclusion.
2. Write the converter in pure array ops (no per-node callables).
3. Build a tiny synthetic workspace using that class, evaluate it in the native
   reference, and compare value (target ~1e-14) and gradient (~1e-11) at several points,
   including near every branch boundary.
4. Add regression tests for value, gradient and Hessian.

Matching numerics sometimes needs the reference's rounding: e.g. keep ROOT/Cephes erf's
two subtractions in negative tails (use optimization barriers so XLA does not fold them).

## 3. Branches and invalid domains
- Use masked branches: replace inactive operands with finite safe values, evaluate,
  zero the inactive result. Otherwise NaN from the dead branch leaks into the gradient
  (`where` alone is not enough).
- Use `lax.cond` / `lax.map` where short-circuiting matters for derivatives.
- Finite differences across a discontinuity are meaningless; test each side separately.

## 4. Validating gradients on the full model
Use several independent checks; none alone is enough.
| Check | What it proves |
|---|---|
| Value vs native at nominal + N random points | the function is right |
| AD gradient vs native finite differences, all parameters | the gate |
| Forward-mode AD vs reverse-mode (bypasses custom VJPs) | your custom VJPs |
| CPU vs GPU, sequential vs batched | no device/batching bug |
| Independent NumPy re-evaluation of components | no shared-code blind spot |

## 5. When AD and native finite differences disagree
Assume nothing; find out which side is wrong.
1. **Is the native FD itself stable?** Recompute it at many step scales (1/8 … 64×)
   with standard and half steps. Accept a native value only if ≥3 resolutions agree
   (e.g. within 1e-5 relative). Near a minimum (Asimov), gradients ~1e-12 are
   dominated by cancellation; relative gates there are fragile.
   Stability is necessary, not sufficient: audit the **entire stencil** against
   clipping and interpolation branches. Several large steps can all cross a kink
   and agree on a biased plateau. For example, at x=-1e-6 the derivative of
   `x + 0.001*max(x,0)` is 1, but crossing stencils can stably return about 1.0005.
   Preserve that failed comparison; establish the branch from native source/cache
   evidence and use an independent native analytic score or explicitly documented
   branch-local diagnostic. Never select a new step by agreement with AD.
2. **Use a higher-order stencil** (order 8 vs order 4) on the unresolved columns only.
3. **Derive an analytic native score** for that parameter (e.g. a location parameter's
   derivative via the density's x-derivative) and evaluate it natively. This settled
   `CMS_hmm_peak_cat0_ttHlep` at all 26 points.
4. Reuse native caches but only copy what the parameter depends on, after a precheck
   proving identical results, so these diagnostics stay affordable on huge models.
   A memoized PDF row array and its object's internal cache need not describe the
   same parameter point. Test a return to a previously evaluated point. If an
   internal cache must be refreshed through the public native evaluator, record
   the extra calls and require the full NLL and original score arrays to remain
   unchanged. Reading a workspace clone is not evidence about the optimized
   object actually used by the NLL.
   Precheck accessors in the pinned ROOT version, including fixed RooRealVar
   coefficients as well as RooConstVar. Older PyROOT can expose a derived object
   through its declared base proxy: `InheritsFrom` succeeds while derived methods
   are unavailable. Use checked C++ dynamic casts before reading derived fields;
   newer PyROOT's automatic downcasts can conceal this failure in local tests.
   When enumerating parameter leaves, include a directly bound RooRealVar itself:
   `getVariables()` can enumerate only its servers. Assert the expected parameter
   names/types in fixtures, so passing comparisons cannot hide omitted coverage.
   Activating native analytic BB profiling can leave a fixture's constructor
   row cache unprofiled until a distinct native parameter evaluation. Audit
   initialization separately: preserve the initial NLL/cache, record any warmup
   and its NLL change, and require stable repeated values afterward. Never
   silently warm up or replace an existing full-model reference to pass a gate.
   A member named like an enable flag is not necessarily authoritative: in v10
   CMSHistSum the copied `analytic_bb_` field is uninitialized; `bb_.init` records
   the actual bound BB state. Verify readers against pinned source and runtime.
   Also inspect active collections: serialized RooParametricHist objects can
   have `_hasMorphs=true` with an empty `_coeffList`. Native evaluation returns
   the identity factor without reading smoothing tables. A diagnostic must not
   reject that supported empty path solely from the flag; test empty and
   nonempty cases separately, including unused invalid smoothing metadata.
5. If a valid, branch-local native derivative is stable and still disagrees,
   bisect the port by channel, then by node. A score with an analytic outer NLL
   derivative but finite-differenced cached inputs still needs these checks on
   its inner stencils; it is not fully analytic.

For an NLL mismatch appearing only at a fitted endpoint, compare independent
NumPy components with the actual native cached terms before blaming XLA. A
stable analytic expression can differ from a cancellation-prone native formula:
v10's vectorized exponential uses `(exp(a*high)-exp(a*low))/a` for every nonzero
slope. Near `a=-1.56e-10`, a 4.16e-9 relative normalization error explained a
1.72e-6 channel NLL difference. Prove the cause by reproducing the isolated
native channel and changing only that normalization in an offline calculation.
Keep the failed gate and the unchanged native reference; an explanation does
not authorize a tolerance change or silently changing either evaluator.

Record each unresolved column by name; "8/12 fail" with names beats a looser tolerance.
A fail-fast reader's first unsupported class is not a complete dependency census.
Before claiming that a new rule closes a parameter's score, enumerate all active
channels/functions and direct constraints for that parameter. A repaired scalar
normalization can still leave dependent channels outside a histogram reader's
scope. Keep component checks distinct from a complete parameter score.


## 6. Scaling the native reference
A full native NLL can take 0.1–3 s per call and the gradient needs ~2N calls. Shard
FD columns across batch jobs, stream progress (RSS, wall time), and merge shards with
a check that the shards reproduce the monolithic NLL.

Bound reference artifacts during generation, not only after the job exits.
Large score records can duplicate the same channel evidence in a progress stream
and a final in-memory list, exceeding the scheduler's transfer cap and losing the
entire result. Write lossless, independently closed compressed records with a
checksum manifest; keep only counters in the batch summary. Enforce the output
budget before committing each record and preserve completed records on failure.
Audit large record streams on the compute node, copying back small comparisons.
Partial capture counts and successful transport are not validated derivatives:
retain missing/unsupported parameter names and the original numerical gates.
Validate worker checksums in an isolated directory containing exactly the
transferred files. A successful submission-host check or Condor dry run can
miss a manifest that names untransferred submission scripts. Keep staging
checks separate from worker payload checks, and preserve any failed-job receipt.
Apply that discipline to downstream audits too: repeated dependency inventories
can inflate a small compressed capture into a huge JSON report. Keep the full
checksum-bound records remotely; summarize inventories by counts, classes and a
canonical checksum without changing score comparisons or hiding unavailable
columns. Check artifact size before copying it back to a constrained machine.

## Anti-patterns seen
- Trusting a single default-step FD as truth.
- Treating cross-device agreement as reference validation.
- Approximating an unsupported class with a "close enough" one.
- Fixing a mismatch by changing steps or tolerances.
