# combine-jax design

The target is a **Combine-equivalent statistical tool**: `-M` style method dispatch
over a general JAX likelihood, with native Combine v11 as the numerical reference.
The likelihood, neutral importer, float64 AD and native-agreement gates remain the
foundation. The historical VV prototype supplies evidence about conventions and
formulae, not a channel-specific implementation to wrap. `STATUS.md` distinguishes
implemented and validated functionality from this design; naming a method here
does not claim that it is implemented or equivalent.

## Statistical method layer

### End-to-end blinded runtime and DHI integration

`combine-jax limits` is the deployable, pre-fit expected-only entrypoint. The
original `limit`/`-M AsymptoticLimits` interface remains available for prepared
models and research workflows. The deployable entrypoint accepts one POI,
mass, assignments, ranges, freeze patterns/groups and explicit discrete policy.
It rejects observed operation and expected mode without `--noFitAsimov`.
Unknown parameters and unsupported configurations fail rather than approximate.

Three isolated processes own the three environments: native v10.3.3/ROOT 6.30
generates background-only nominal Asimov rows; the public source-built v11/pip
ROOT 6.40 importer exports neutral arrays with `v10_density`; CUDA LCG JAX 0.10.1
loads only those arrays. Version checks and dataset/source hashes bind the stages.
No ROOT interpreter is loaded in the fit process. The output has five native
quantile labels in a genuine `limit` TTree, no observed row, and a required JSON
receipt. A returned midpoint without an evaluated bracket or an accepted NLL
crossing is explicitly unresolved and stored as NaN. A failed fit aborts output.

The runtime installer builds the importer from pinned public source, packages
relocatable dependencies and code, and records an archive checksum. DHI's opt-in
`limit_backend=jax` uses a separate output suffix, adds a runtime-bundle dependency
and H100/A100 request through its HTCondor hooks, and bootstraps worker scratch.
Native task commands and paths retain their defaults; MergeUpperLimits and plot
loaders consume the standard tree unchanged. The native CMSSW path is configured,
not embedded in the wrapper. Every GPU run keeps float64, structural-sequential
gradients, the 8 GiB static guard and process/GPU memory measurements.

The law integration is validated independently of the older artifact-driven
benchmarks: it starts from CreateWorkspace, generates and imports its own Asimov,
and compares the merged expected NPZ with native v10. No reference value is a
runtime input. Statistical bracketing and convergence thresholds are unchanged.

Method dispatch accepts workspace/datacard inputs and an explicit POI, configuration,
dataset and method options. It separates ROOT import and dataset preparation from
the reusable device likelihood, fit and conditional-profile executables. Each result
records method/dataset identity, immutable inputs and source versions, start/bounds,
optimizer status and evaluation counts, numerical procedure and tolerances, backend,
compile/execution wall time and memory. Failed fits or crossings remain failures,
not accepted intervals or speed claims. Existing likelihood classes and the old
backend remain intact as the oracle.

The first method is **AsymptoticLimits**, with expected/blind operation only.
**MultiDimFit** will expose best fits and 1D/2D likelihood scans; **FitDiagnostics**
will provide covariance/Hessian and profile-based uncertainties; **Impacts** will
reuse conditional fits and validated profile plans. **HybridNew/toys** is later
work requiring separate ensemble and sampling validation. Every method is checked
against native Combine v11 before claims of equivalence, including each reported
quantile for limits. Existing fitted-point and gradient agreement alone does not
validate a statistical method.

## AsymptoticLimits conventions (v11 source audit)

The reference is the bundled v11
[`AsymptoticLimits.cc`](vendor/combine_pypi/vendor/CombinedLimit/src/AsymptoticLimits.cc)
and [`AsimovUtils.cc`](vendor/combine_pypi/vendor/CombinedLimit/src/AsimovUtils.cc).
The method defaults are `rule=CLs`, `qtilde=true`, `newExpected=true`,
`signalStrengthForExpected=0`, `minosAlgo=stepping`, `rAbsAcc=0.0005`,
`rRelAcc=0.005`, and no strict upper bound unless `--strictBounds` is set.
The global confidence level is 0.95. The new tool exposes expected/blind modes;
native Combine's default `--run both` is deliberately not an enabled default.

Native **`--run blind` forces pre-fit Asimov generation** (`noFitAsimov=true`).
Native **`--run expected` normally performs a conditional background-only fit**
to its input dataset, generates at r=0 with those nuisance values, and constructs
matching global observables. `--noFitAsimov` skips that conditional fit and uses
the nominal workspace nuisance/global values. For HHVV, conditional expected mode
must explicitly identify an Asimov input; it never fits observed data implicitly.
Generation and global-observable snapshots must follow `AsimovUtils`, including
Gaussian, Poisson and supported non-Gaussian constraint conventions. Combine v11/ROOT 6.40
may export the dataset/neutral buffers; it must stay isolated from LCG CUDA ROOT.

For a nonstandard POI name, native `GenerateOnly --expectSignal 0` is insufficient:
`Combine.cc` applies that option only to a POI literally named `r`. A generation
control for this model must explicitly set `r_hhvv=0`. Our method calls
`asimovDatasetNominal` with the ModelConfig POI value zero directly. Generation
provenance must record the actual POI value and canonical row identity, not infer
background-only data from a filename or command option. The noVH MX4000 control
finds identical v11 CLI/direct rows but different v10.3.3/ROOT 6.30.07 rows on the
same workspace. A v10 reference therefore needs an explicit stack/data comparison;
its use must be explicit in method validation and artifact provenance.

The October 23 fixed-point investigation identifies a concrete compatibility defect
for the VV4q variable-bin templates (see
[the complete audit](reports/asymptotic_vv4q_stack_localization_audit.json)).
`FastVerticalInterpHistPdf` and `FastVerticalInterpHistPdf3D` construct TH1/TH3
caches with `createHistogram` and normalize the heights using bin widths. ROOT 6.30
scales the sampled density by a constant average bin volume, which cancels during
normalization. ROOT 6.40 scales each bin by its actual volume, so the same Combine
code instead normalizes bin masses as density heights. The independent stored-bin
reconstruction supports v10's density interpretation for these inputs. The NLL
legacy and CPU evaluation backends retain this change; switching backend is not a fix.

Asimov preparation is affected independently: newer ROOT retains density scaling for
extended `createHistogram`, whereas the old implementation disables it. Combine's
`ToyMCSamplerOpt` then applies its own width normalization and volume factor. Both
versions' actual Asimov weights are reconstructed exactly from the saved densities,
bin volumes and native TH3F rounding. Therefore v11's generated dataset is not the
Asimov of its normalized PDF on these grids. This also means agreement with a literal
v11 evaluator does not establish correct variable-bin model semantics.

The plain RooHistPdf case must be distinguished from a legacy morpher wrapping it.
The user's v10-built 1D/2D cards have unchanged RooHistPdf and mixture densities
and same-data NLLs across v10, official v11 and pip v11 (26 points each, maximum
absolute difference 4.55e-13). Direct RooFit `generateBinned(ExpectedData)` is also
correct. Only Combine's Asimov path adds the extra volume there. Conversely, the
shared-data 3D morpher control changes nominal NLL by +20.4873576673; summing
`-weight * log(pdf_v11/pdf_v10)` reconstructs that change within 3.56e-14.
All constituent RooHistPdf densities still agree exactly. This isolates the cache
construction, not the scalar RooHistPdf or NLL evaluator. See the
[shared-input controls](reports/variable_bin_paths_comparison.json).

Source hashes of `AsimovUtils.cc`, `ToyMCSamplerOpt.cc`, `VerticalInterpHistPdf.cc`,
`FastTemplate.h` and `FastTemplate.hpp` are identical in all three installations.
There is no v10-to-v11 change in those Combine files. The ROOT changes are
[2f06c041](https://github.com/root-project/root/commit/2f06c041f99faa31c760b149b236b31e2d740910)
(average volume replaced by each bin's volume in `fillHistogram`) and
[16b33f63](https://github.com/root-project/root/commit/16b33f63e4f3d79df980be764e571f3b0a5817c5)
(extended `createHistogram` no longer overrides scaling to false), both committed
2025-02-17. The latter is documented in the
[ROOT 6.36 release notes](https://root.cern/doc/v636/release-notes.html#fix-for-rooabsrealcreatehistogram-with-extended-pdfs).
Use these identified changes and measured versions, not an unverified ROOT>=6.32
threshold. An Asimov-only fix is sufficient for the plain RooHistPdf controls;
VV4q also requires the already-validated morpher-cache correction.

The user has now selected the v10.3.3 / ROOT 6.30 density convention as the
reference for VV4q and affected variable-bin 2D/3D templates. The official v11 /
ROOT 6.36.07 26-point probe is bit-identical to pip v11 / ROOT 6.40, ruling out
a pip-only defect. New imports must record an explicit `v10_density` convention;
`native` retains literal installed-stack behavior for the old oracle and audits.
This policy corrects both morph-cache construction and Asimov row generation;
it does not change the interpolation, floors, fit acceptance or statistical method.
The same 1e-6 NLL, 1e-4 gradient and 1% expected-quantile gates apply against the
selected reference. Existing artifacts retain their original bytes and provenance.

Implementation: tensor imports default to `template_convention="v10_density"`;
the old graph backend/default registry retains `native`. The CLI exposes
`--template-convention` for workspace preparation, and prepared artifacts cannot
change convention at execution. Copied legacy 1D/2D/3D morpher caches are rebuilt
from order-zero RooHistPdf densities, with the old constant-volume histogram
conversion and unchanged Combine normalization/morph/floor arithmetic. Direct
stored FastVerticalInterpHistPdf2/2D2 arrays and RooHistPdf density extraction remain
unchanged. Mismatched template/observable grids are rejected. Only the native
TH*F import boundary retains v10's historical rounding; model buffers, templates,
gradients and fitting remain float64. No vendor source or original workspace is modified.

For pre-fit Asimov preparation, native AsimovUtils still establishes globals and
the grid. Variable-grid channels are reweighted using corrected densities and the
v10 ToyMCSamplerOpt histogram recipe: extended density, width normalization, then
exactly one bin volume. Complete unique bin-center grids are checked explicitly;
pseudo-Asimov rows are refused. Every synthetic regression weight and every noVH
MX4000 weight agrees exactly with the saved v10 generation. Conditional post-fit
expected mode on affected variable grids is explicitly unsupported until its v10
background-fit/global-observable gate is added; `blind` and `expected --noFitAsimov`
are the validated generation paths. Generic unaffected conditional modes retain
the existing v11 same-point and BB-lite gates.

For a tested nonnegative mu, the one-sided statistic is zero if mu_hat > mu.
Otherwise it is twice the profiled NLL difference against the constrained global
minimum (mu >= 0), equivalent to using the mu=0 conditional denominator when the
unconstrained estimate is negative. For asymptotic calculations, let a=sqrt(q_mu,A).
For q <= a^2, CLs+b=1-Phi(sqrt(q)) and CLb=Phi(a-sqrt(q)); for q > a^2,
CLs+b=1-Phi((q+a^2)/(2a)) and CLb=Phi((a^2-q)/(2a)). CLs is their ratio.
Zero-sensitivity/degenerate cases need explicit handling rather than NaN limits.

Combine reports quantile labels **0.025, 0.16, 0.50, 0.84, 0.975**, rather than
the exact Gaussian probabilities at integer sigma. For each p, N=Phi^-1(p) and
the default `newExpected` procedure solves the background-Asimov profile crossing
`NLL_A(mu)-NLL_A(min) = 0.5 * [N + Phi^-1(1-(1-CL)*p)]^2`.
It does not derive all bands from one constant fitted sigma. This crossing is
the asymptotic CLs=1-CL condition; the source states the q_mu and q_mu-tilde
expected-limit expressions coincide. Report the median and all four bands.

The reference median starts at 5% of the initial rMax and uses the `stepping`
algorithm: retain nuisance warm starts, halve strides after overshoots and when
approaching the threshold, stop when the stride error falls below
`max(rRelAcc*mu, rAbsAcc)` or the absolute NLL crossing residual is below
`0.05 * conditional_minimizer_tolerance`. The default nonmedian bands switch to
bisection: brackets are [0, median], [lower band, median], [median, median+2*sigma],
and [previous band, median+4*sigma], with sigma=median/Phi^-1(0.975).
Bisection uses the same relative/absolute accuracy and NLL residual criterion.
Without strict bounds, a crossing at/above the current rMax extends it to 1.1*mu;
rMax is an initial search scale, not an immutable physics cutoff. Strict bounds
cap the search/output at rMax. Additional watchdogs must report an unresolved
crossing explicitly; they may not manufacture a saturated limit in non-strict mode.
Alternative native `minos`/`new` algorithms and nonzero expected signal are outside
the first implementation and must be rejected until independently validated.

Conditional fits reuse the existing fit/profile implementation with the
`structural_sequential` gradient and preceding successful nuisance solution as
the next start. Convergence and native method comparison remain separate gates;
matching the CLs formula cannot excuse failed profiling. Validation proceeds from
generic counting/autoMCStats/parametric cards (all five quantiles within 1%) to
the user's independent VV4q dhi references, then full HHVV. HHVV remains blinded;
GPU execution is HTCondor H100/A100 only, never lxplus905.

AsymptoticLimits inherits `LimitAlgo`, so it does **not** call the
`FitterAlgoBase::optimizeBounds` pass used by the earlier fit benchmark.
The method importer preserves workspace nuisance ranges for both background
preparation and subsequent profiles. Earlier artifacts with removed nuisance
bounds must be reimported; the likelihood and ordinary fit policy are unchanged.

Conditional Asimov construction depends on the actual background-fit endpoint,
not just whether it passed the EDM criterion. The synthetic autoMCStats audit
crosses two preparation engines with two profile engines: native/JAX limits agree
to about 1e-8 on either shared generated dataset, while distinct stopping points
change bin weights and globals and shift returned crossings. Native generation
at a common point preserves BB-lite parameters and Gaussian/Poisson globals;
there is no measured BB/global-observable rule mismatch in that case. The former
0.534% difference is therefore not treated as unexplained tolerance slack.
The separately corrected bound convention changes the optimizer path, but does
not make finite-tolerance preparation endpoints identical. No stopping or
crossing tolerance is tuned to the reference. Shared Asimov input comparisons
isolate the likelihood/profile method from this preparation sensitivity.

The first implementation exposes `combine-jax -M AsymptoticLimits --workspace ...
--datacard ... --poi r --run blind --output result.json` (`limit` is an alias).
Preparation in the isolated v11 stack writes a new workspace and framework-neutral
background-Asimov model, never modifying the original. Conditional nuisance fits use
JAX; one native NLL evaluation restores analytically profiled bin nuisances and checks
the same-point NLL before native Asimov generation constructs rows and globals.
The native generator is called with its own fitting disabled. A prepared `--model`
can execute without ROOT, also through `benchmark_tensor.py --operation limit`.
Legacy nominal-signal benchmark artifacts are not background-only limit inputs.
The workspace default is `--run blind`; loading a prepared model without `--run`
uses its recorded preparation mode. Explicit conflicting modes, workspace-only
configuration options on prepared models, identity mismatches and observed modes
are errors. The default method rejects active RooMultiPdf models explicitly: the
likelihood's selected discrete choice is not a substitute for native discrete
profiling. This restriction does not change the old likelihood oracle.

The exploratory combined-HHVV GPU worker can explicitly request
`run_expected_model(..., discrete_policy='fixed')`. Both the preparation record
and execution must opt in; the recorded branch indices, selected PDFs and
corrections must match the importer's complete `discrete_choices` metadata.
The result is labelled `exploratory_fixed_discrete_pending_native_validation`.
It is conditional on those branches and is not a native discrete envelope.
The CLI and benchmark defaults still reject these models. This opt-in changes
neither the likelihood, numerical tolerances, nuisance bounds nor old backend.

`LimitProfiler` reuses the existing device profile driver, compiling global and
conditional fits plus a first unprofiled evaluation once. Conditional fits use raw
EDM < 1e-4 (tolerance 0.1, Up=0.5); the native expected global minimum instead uses
Up equal to the median NLL crossing, so its raw threshold is `0.002*0.1*Up`.
An unchanged 8 GiB static guard checks each executable. Nuisance vectors and fit
iterations remain on device; scalar crossing control is Python and receives fit
status/NLL/counts, not per-row likelihood contributions. No full Hessian is needed.
Compilation is reused across crossings within one process; persistent executable
reuse across batch jobs remains a separate unfinished requirement.

All five quantiles pass the 1% gate for counting, autoMCStats and parametric examples
in blind, conditional expected and explicit pre-fit modes (45 comparisons; largest
error 0.534%). This establishes those tested cases only. Single-run CPU timings,
source/input hashes and native results are in
[`asymptotic_generic_validation.json`](reports/asymptotic_generic_validation.json).
Native startup-inclusive wall time is compared to the complete JAX process; JAX
preparation, lowering/compilation and limit execution are also reported separately.
VV4q and full-HHVV method validation and GPU limit timing remain pending.

## Tensor architecture after the structural census

The original graph registry and evaluator below remain the reference oracle. The
new execution path consumes a framework-neutral tensor artifact. ROOT is confined
to import and reference measurements; JAX and PyTorch are execution adapters.
The census found 200 exact channel class trees, 149 singletons, and 14 coarse
density-operator families. Consequently, grouping whole RooFit channel trees is
insufficient: scalar rates, shape parameters, templates, analytic families and
constraints must each become batches with explicit shared-parameter gathers.

The artifact uses versioned JSON metadata plus numeric NPZ buffers, without
pickle, ROOT objects or framework arrays. Transport checksums protect metadata
and arrays. Its compilation structure key describes the operation schedule and
buffer shapes/dtypes, excluding numeric contents, labels and provenance. Current
keys are shape-specialized; future padding buckets, persistent compilation caches
and/or AOT exports must establish reuse across batch jobs. Merely adding this key
does not implement persistent executable reuse or prove flat compile time.

The first implemented layer is an observable-independent scalar program. A
dependency pass distinguishes floating parameters, observables and analytically
profiled nuisances. Fixed-bound parameters become immutable import configuration.
Certified scalar subgraphs with none of those dynamic dependencies are evaluated
once by native ROOT and stored in buffers. Unsupported dependencies cannot hide
inside a folded expression. Observable-dependent fixed PDFs require their exact
data/integration grids before folding; that extension remains separate work.

Remaining scalar formulas lower to a shared arithmetic DAG, then pack by operation
and dependency depth. Ragged sums/products use neutral padding masks. Runtime
gathers retain shared parameter and intermediate identities. JAX executes vectors
of arithmetic terms per batch; it does not trace a callback per RooFit node. All
numeric model buffers are function arguments. NumPy executes the same serialized
operation semantics independently of JAX; the optional PyTorch counting/template
adapter also consumes the same layout. The scalar-only workspace+card boundary explicitly lists pending
PDF/constraint roots; a separate complete importer now handles counting/templates.

Full HHVV scalar values are checked against native Combine at the existing 26
points. This gate is not a full likelihood or derivative validation. Fixed-bound
coordinates must retain their imported values; changing them invalidates the
import. Their new derivatives are zero by construction, so derivative checks use
the free coordinates. The scalar-only program still rejects dynamic conditionals
and floating splines. The coordinate-bound program below now lowers conditional
branches safely; this machinery has not yet been generalized to the scalar-only
layer. Frozen mass splines already fold natively.

Constraints now stack by semantic type, with gathers into scalar outputs and an
explicit vector of analytically profiled nuisance values. Shared constraint names
occur once. Fixed raw integrals, observed Poisson log-factorials and wholly fixed
penalties fold at import. Ordinary and fast Gaussian/Poisson terms retain separate
normalization/rounding rules and the fast large-count branch. Generic dynamic
observed-value derivatives of incomplete-gamma mean normalizations still require
a later operator and raise an explicit error. HHVV uses two batches: Gaussian
and Poisson, with all 498 Poisson normalization integrals fixed.

Uniform-mixture channels gather padded rates and retain observed weights and
finite observable volumes. Fully fixed or masked channels become runtime buffers.
The complete counting/template importer checks workspace/card channel membership and
refuses every unimplemented active channel shape. A composite JSON/NPZ artifact
contains scalar, constraint and channel schedules and buffers; it can be inspected
and evaluated with NumPy without importing a device framework. Its JAX adapter
performs one objective evaluation with device reductions and no closed-over model
arrays. Complete counting NLL/gradient/Hessian gates and HHVV's separate constraint
and counting subsets pass. The earlier native-supplied profile test remains as
evidence for constraint consumers; the new template operator computes profiles.

Aligned CMSHistFunc sums and CMSHistErrorPropagator channels pack into histogram,
process, bin and observation axes. Morph batches group by additive/log mode, with
neutral padding and shared histogram gathers. Import folds fixed morph terms and
fully fixed histograms, and encodes native rebinning as a width-weighted projection.
The runtime retains active-bin normalization, stored tails and morph floors.
All type-0 through type-4 bin corrections operate on tensors. The stable quadratic
profile solve uses native last-row data assignment, global observations and bounds;
invalid bounds or zero-error profiles produce NaN rather than clipping or freezing.
The certified extractor validates profile/constraint associations, while packing
rechecks geometry, data, bounds and globals. Profiled propagators currently require
a fixed unit outer coefficient; other cases are explicit unsupported errors.

One objective evaluates scalars, template channels and profiles, then constraints
and counting terms, with all reductions on device. NumPy independently executes
the same neutral artifact. Full autoMCStats NLL/gradient/Hessian gates pass and
HHVV checks cover all 182 template channels and 74 profiles at 26 native points.
That leaves 99 HHVV channels requiring further operators. Same-structure channel
replication has constant traced operator count; this is not a measured guarantee
of flat compilation time or batch-job executable reuse.

The fast-histogram layer packs all five FastVerticalInterpHistPdf classes by
normalization-grid layout and additive/log mode. Each batch evaluates two matrix
products against stored difference/sum templates, then normalizes along the native
slice axes. Conditional grids omit their first axis from normalization exactly as
stored by Combine. Inactive 1D tails bypass normalization. Fixed morph contributions
and fully fixed histograms become import buffers. All grids stay complete even
when the data occupy only a subset of bins.

A separate neutral query artifact maps actual coordinates to native `(low, high]`
bins. It represents full or partial integrals as weighted gathers using geometric
overlaps with imported default bounds; those weights are distinct from conditional
normalization weights. Fixed queries fold too. NumPy and JAX adapters consume the
same numeric buffers, without runtime RooFit graph walks or array closures. This
component is validated independently before composition into the remaining channels.

Binned channel composition resolves `(PDF, normalization axes, value/integral/density)`
requests during import. Mixtures normalize yields or retain the implicit final
fraction. Product factors preserve nset/cset ownership, dropping unrequested factors
and rejecting overlapping or cyclic normalization. Vertical interpolation separately
combines raw component values and advertised integrals, including floor ordering.
Floating 2D bins preserve x-fast parameter indexing and stored volumes; step functions
retain their advertised unit integral even when the last bin is floored. Frozen
MultiPdf choices and channel corrections remain outside AD.

Fixed queries precompute into arrays. Dynamic operations group by dependency depth,
operator and data-row count with padded child/coefficient axes. A shared work buffer
and explicit index maps connect these batches; no per-node callable enters the JAX
trace. The validator certifies unique fast bindings, density outputs and channel
outputs before function construction. Scatter-set kernels declare that guarantee
to JAX, avoiding reverse-mode duplicate-destination bookkeeping. Runtime index
arrays must obey the validated artifact contract.
The complete workspace+card importer now composes supported binned channels
with counting, CMS templates, profiles and constraints into one neutral artifact and
one device objective. Analytic queries use the same schedule and runtime scalar
gathers. Gaussian, exponential, power, DoubleCB and ErfExp shapes retain their
finite-range analytic integrals, signed tails, logarithmic/zero-slope branches and
native error-function rounding. Bernstein bases and integral weights precompute
into runtime arrays. Entire fixed queries fold before tracing. No integration-grid
approximation is introduced.

Observable-dependent shape parameters use a coordinate-expression program. Each
analytic query keeps only its actual observable dependencies, forms unique joint
coordinate rows and stores an exact gather back to the original data rows. An
analytic normalization query excludes the integrated coordinate and evaluates the
advertised primitive at its bounds. Dependence of shape parameters on that
integration variable still raises a named error. Full histogram normalization
grids are unaffected by this compression.

Fixed expression subtrees and whole fixed shapes precompute into arrays. Dynamic
expressions batch by opcode, dependency depth and row layout; they share scalar
parameter gathers. Each conditional branch carries an activity mask. Before an
operation executes, inactive operands are replaced with finite safe values, then
its inactive result is zeroed. This preserves selected-branch derivatives without
evaluating invalid inactive sqrt/log/division arguments. The program holds no
per-node callables or framework objects. The initial complete HHVV artifact used
216,540 grid rows for 192 conditional queries instead of 8,472,000 repeated query
rows; 40 queries folded fully.

Extended shape channels now retain only positive-weight observed rows, padded to
power-of-two sizes by repeating a retained coordinate with zero weight. Original
coordinates and weights are validated before selection. Source-row indices and
padding masks remain in the neutral artifact for auditing; an importer option
retains the original rows for comparison. Empty observations retain the extended
rate term. This transformation does not touch CMS template/profile inputs, full
histogram normalization grids, observable ranges or analytic primitives.
All analytic queries, including scalar-parameter shapes, use exact unique
coordinate gathers with power-of-two padding. Padded child axes have explicit
masks. Work-buffer construction uses NumPy chunks instead of Python scalar lists.
Padding limits layout proliferation. Separate scaling and executable-cache
experiments are summarized in STATUS.md.

Negated RooExponential is also an explicit unsupported case: under the checked v11
flags its optimized cache ignores the negation, while direct RooFit evaluation
honors it. A saved native probe records the disagreement; the importer does not
silently select either convention. The old backend remains unchanged.

The complete float64 objective and gradient now pass the saved nominal plus
25 random native reference gates. Gradient capture uses one JIT with runtime
model arrays and device reductions; the unchanged nominal, directional and
random finite-difference checkers consume its saved arrays. Fixed-bound entries
have zero tensor derivatives. The Hessian evaluator linearizes the gradient once,
using the NLL as auxiliary output. A `lax.while_loop` applies that linear map to
bounded batches of unit directions and writes the resulting columns into a device
matrix; a partial final batch is padded with zero directions. Parameters, residuals
and blocks stay on device until completion. Float64 and vector rank are checked;
fixed coordinates have zero rows/columns and an empty vector yields an empty matrix.
Generic native-card checks pass; full HHVV curvature validation and device fits
remain pending. Compiler allocation and RSS guards bound the first full HHVV run.
The optimizer loop and parameters then stay on device through convergence, with
an exact final AD Hessian and its inverse. iminuit remains a cross-check. Benchmark
the same model under JAX JIT, eager execution and PyTorch CUDA graphs, including
compile/load, warm derivatives, memory and full-fit times. Native Combine timings
must use the same worker. CERN Condor A100/H100 jobs are the primary GPU target;
lxplus905 T4 remains secondary. No performance equivalence is inferred from its
limited FP64 throughput.

## Workspace boundary and graph

ROOT is used only to load, inspect and export the model and to obtain independent
validation references. Load the installed Combine dictionaries before opening a
file. Preserve input files. Export an immutable graph of named nodes, explicit
ordered dependencies, observable ranges, numeric arrays and class-specific
attributes. Retain original class and node names in diagnostics. Reject cycles,
ambiguous names, unsupported classes and unsupported configurations explicitly.
No fallback to numeric sampling or nominal values for unknown floating nodes.

An exact-class registry owns extraction, JAX value evaluation and analytic
normalisation integrals. Inheritance alone is not evidence of equivalent semantics.
Composite nodes specify their child normalisation sets; overlapping or conditional
products require explicit treatment. Histograms preserve bin edges, cache storage
order, interpolation, floors and width-weighted normalisation. Formulas use a small
AST allowlist, never Python `eval` on workspace text. Fixed-node folding must retain
provenance and must not hide unsupported dependencies.

Continuous parameters have a stable lexicographic order and retain initial values,
limits, errors and POI/nuisance roles. Observables, global observables, constants,
channel masks and discrete categories have distinct bindings. Discrete RooMultiPdf
indices are fixed outside AD; each alternative carries its correction and active
parameter set. Graphs and parameter maps are rebuilt when constants or alternatives
change. No ROOT calls occur during tracing, evaluation, gradients or Hessians.

## Data and likelihood conventions

Bind actual dataset coordinates and weights, whether the storage is RooDataSet or
RooDataHist. A weighted RooDataSet can contain histogram rows or unbinned events;
never infer physical binning from that class alone. Keep zero and fractional weights.
Detect observed/Asimov geometry differences. Preserve channel categories and reject
unsupported weight conventions explicitly.

For a channel with densities p_k, yields nu_k, intensity I_i = sum_k nu_k p_k(x_i),
N = sum_k nu_k and W = sum_i w_i, the canonical extended objective is

    -sum_i w_i log(I_i / N) + (N-W) - W (log(N)-log(W)).

Use the W=0 continuous convention. Analytic PDFs use continuous finite-range
normalisation even for bin-centre rows. Do not replace N by midpoint quadrature.
RooRealSumPdf requires its function integrals and Combine's cached-integral rules.
Separate the canonical objective from Combine's constant zero point and explicit
reference-point offset. Record both conventions in each validation artifact; an
offset must never mask a parameter-dependent discrepancy.

Factor constraints out of channel products and deduplicate shared constraints.
Match fast self-normalised Gaussian/Poisson constraints separately from generic
RooFit constraint normalisation. Preserve global observables and conditional sets.
Apply channel masks to channel terms, with constraints and mask offsets following
CachingSimNLL. autoMCStats requires CMSHistErrorPropagator's analytic
Barlow–Beeston minimisation and boundaries before evaluating both channels and
constraints. Until implemented, reject these configurations rather than dropping
their terms. Invalid-PDF/error-wall behavior must be declared and tested.

The implemented bin-statistics path exports the stored CMSHistErrorPropagator bin
types and parameters. Type 1 shifts the summed prediction by the coefficient-scaled
total error; types 2 and 3 apply process-level Poisson multipliers and Gaussian
error shifts. Analytic profiling is optional and follows the v11 quadratic solve
for unfrozen type-1 nuisances. Compute those solutions once per evaluation, use
them in both channel and constraint contexts, and remove them from the continuous
parameter vector. AD through the solution includes its full curvature; tests compare
the resulting Hessian to the Schur complement of the explicit-nuisance Hessian.
Frozen bin nuisances retain their saved values. Nonstandard constraints and shared
profile nuisances are rejected. Out-of-range solutions are invalid (NaN in the JAX
function), matching the valid domain of ROOT 6.40's throwing setVal rather than
silently imposing a clipped minimum.
Zero total error for a profiled bin is also invalid: Combine leaves its previous
nuisance value unchanged in that case, so a stateless function cannot reproduce
arbitrary evaluation histories. The explicit-nuisance mode remains well defined.

Ordinary RooGaussian and RooPoisson constraints are normalised over their relevant
ModelConfig nuisance variables. SimpleGaussianConstraint and SimplePoissonConstraint
use their distinct fast-log forms. Registry log-value callbacks avoid probability
underflow and preserve the fast Poisson large-count convention. FormulaVar uses an
allowlisted AST interpreted by JAX primitives; unsupported syntax is a named error.

CMSHistFunc exports its fine-grid templates and vertical-morph settings directly.
When Combine enables automatic rebinning, the graph also retains its stored
fine-to-coarse assignment and both geometries. Morphing, log-mode normalization
and the fine-bin floor precede width-weighted accumulation into the output bins.
Nonaligned binnings follow Combine's whole-bin midpoint assignment. A reduced
fine-grid active range contributes only its active bins; empty output bins remain
zero. Values, analytic integrals and CMSHistErrorPropagator use the resulting
output geometry and the error array exported by Combine. Horizontal/external
morphs and nonzero morph strategies still raise explicit unsupported errors.

VerticalInterpPdf interpolates raw child functions and their advertised analytic
integrals separately. Its nonpositive-value replacement floor is distinct from a
minimum clamp. The additive algorithms are supported; multiplicative or forced
numerical integrals fail explicitly. FastVerticalInterpHistPdf2 instead exports
stored nominal/sum/difference arrays and normalises the morphed active bins by
their widths. Inactive cache values stay fixed, and its self-normalised density
must not be divided by the integral again. Exact-bin quadrature checks the integral
independently against ROOT values, including unequal widths and inactive tails.
The older one-dimensional fast class uses the same JAX kernel, with nominal and
morph arrays read from its own initialized copy. This preserves the class's
histogram-building behavior and avoids a v11 conversion constructor incompatible
with ROOT 6.40. Only fixed RooHistPdf components are currently accepted there.
RooHistPdf exports its stored grid and direct observable mapping. Its bin edges
use the opposite interior-edge convention from FastHisto; the uppermost edge stays
in the last bin. Negative values are clamped in point evaluation while the full
analytic integral retains the signed weights, as in ROOT. Unsupported histogram
interpolation and partial integral modes fail explicitly.

Multidimensional fast morphers export their stored geometry through ROOT's own
dictionary serialization with 17-digit doubles, preserving private axis edges and
normalisation weights. Temporary copies expose nominal and morph arrays without
modifying input workspaces. Conditional templates normalise each fixed-x slice
over the other axes; their stored weights omit x width. Full and partial JAX
integrals use actual geometric overlap widths, so integrating a conditional
template over all axes need not return one. The density retains the class's
selfNormalized behavior. Tests cover empty slices, unequal widths, every nonempty
axis subset, restricted bounds, finite gradients and extended NLLs.

Exponential normalisation uses a stable finite-range antiderivative with a
small-slope series that preserves first and second derivatives at zero. The two
arguments may exchange their value/integration roles; infinite or degenerate
ranges and dependent arguments are rejected. Standard Bernstein PDFs retain
RooFit's reference bounds and power-basis analytic integral. Generic parametric
validation uses separate observable columns for each channel's reference NLL.
Hessian comparisons report both raw differences and differences in units of the
nominal parameter scales; finite differences remain a validation method only.

RooSpline1D exports its stored table and algorithm. Linear and Newton-polynomial
interpolation lower directly; cubic/Akima variants obtain knot derivatives from
the same ROOT/GSL interpolator at export, which uniquely fix each cubic polynomial.
This is exact coefficient extraction, not a sampled surrogate. Evaluation and
analytic integration then use only immutable coefficients and JAX. Out-of-knot
values are NaN, including periodic modes; transformed arguments compose through
the graph but analytic integrals require a direct variable. Fast Bernstein
specializations export their actual conversion matrices and preserve the implicit
first coefficient of one and the full-range v11 integral convention.

RooProdPdf densities retain each factor's explicit normalisation-variable (`nset`)
or conditional-variable (`cset`) specification. Nonempty effective normalisation
sets must be disjoint and the conditional dependency graph acyclic. Factors with
no requested normalisation axes disappear, matching ROOT's constraint cancellation.
An empty effective specification instead reverts to ordinary normalisation; in
particular a cset that excludes every requested axis can require a joint integral.
Overlapping/cyclic cases and nonstandard cutoff/default/reference-range modes are
rejected. Raw values multiply raw child values; raw integrals currently require
disjoint dependencies on integrated axes. Tests compare the installed ROOT 6.40
behavior, including nested products and mixtures. ROOT's
[factorization source](https://root.cern/doc/master/RooProdPdf_8cxx_source.html)
provides additional context; the installed-stack tests define the certified scope.

DoubleCB preserves signed alpha thresholds, absolute-alpha tail coefficients,
the optional x-xp coordinate, and v11's logarithmic tail-integral branch throughout
|n-1| <= 1e-5. Its advertised left/right integrals remain independent when signed
alphas make those intervals overlap; tests certify this convention directly.
Nonpositive widths/powers, zero alphas and nonpositive active tail bases are outside
the current valid domain. ErfExp preserves the width floor and exact zero-slope
substitution of -1e-7 in both value and integral. This substitution is discontinuous;
AD follows the selected branch, so finite differences across zero are not a
meaningful check of that branch derivative. ROOT's
[error-function implementation](https://root.cern/doc/master/TMath_8cxx_source.html)
uses Cephes: retaining the two subtractions in negative tails matters after 1+erf
cancellation. JAX optimisation barriers retain those intermediate roundings.
Formula conditionals parse structurally into the allowlisted AST. Scalar lax.cond
and elementwise lax.map preserve short-circuit derivatives when an inactive branch
would otherwise produce NaN; this may need optimisation for large event arrays.

Floating templates retain their source-class conventions. ParametricHist2D uses
x-fastest bin parameters divided by stored bin areas, excludes the upper edge and
advertises a default joint integral equal to the sum of parameters. StepFunction
computes its final bin from the remaining width-weighted area and replaces a
nonpositive final height with 1e-6, while its default analytic integral remains one.
Neither default integral is replaced by geometric quadrature after changing the
observable bounds. Power integrals use a stable n=-1 limit and support a zero lower
bound when integrable. Recursive fractions follow the stored coefficient ordering.

## Differentiation, fitting and resources

RooMultiPdf category indices and external channel masks are immutable configuration
at export. The selected alternative alone contributes active dependencies; all
source alternatives are still translated to detect unsupported classes. Store the
actual v11 correction and add it according to CachingAddNLL's RooAddPdf flattening.
External masks follow category bin order. Masked channel terms and analytic bin
solves disappear, but independent nuisance constraints remain. Reject a floating
mask control that also influences an ordinary likelihood term. Re-export to change
these discrete settings; no derivatives are assigned to them. Internal dynamic
Combine masking offsets are outside this certified convention.

Data export reads RooDataSet/RooDataHist rows once in C++, routes them by category
index and copies only each channel's needed coordinates and weights. NumPy owns the
immutable exported arrays independently of ROOT. Zero-weight rows and empty channels
are retained, and noncontiguous category indices are resolved by their labels.

Use JAX with float64 on the explicitly selected CPU or CUDA backend.
`Likelihood.local_evaluator()` builds kernels with local
continuous parameter slices and shared immutable templates. Dependency closure
includes the input graphs of any analytic bin profiles. Profiled-bin constraints
are assigned to their owning channel once; other constraints are independent terms.
An optional `constraint_batch_size` groups consecutive independent constraints
into one term with the union of their parameter dependencies. This preserves
constraint ownership while reducing kernel count; the default size is one.
The host API accumulates values, gradients and Hessian blocks into global parameter
order, avoiding bins-by-all-parameters intermediates. Hessian compilations can be
released after each block. Optional `hessian_chunk_size` compiles forward
directional derivatives of each local gradient in fixed-width column batches.
Every basis direction is evaluated and accumulated in global order; the result
is the complete autodiff Hessian, with no finite differences or dropped entries.
The final batch is padded with zero directions and unused output rows are discarded.
The default retains the original dense local Hessian. The original global pure JAX functions remain available
for direct transformations and comparison on small models. Preserve reduced axis
structure for conditional grids. Profile discrete alternatives outside
differentiation. Cache only against all relevant inputs.

`fitting.fit` feeds JAX gradients to iminuit (errordef=0.5) and obtains physical-
coordinate covariance from the free block of the full autodiff Hessian. Nonpositive
or nonfinite curvature has no reported covariance; no pseudoinverse is substituted.
Fixed coordinates have zero covariance, and boundary positions and Minuit validity
are separate fields. `fitting.profile` holds the specified parameter fixed and
reoptimises other active parameters, preserving the baseline's other fixed values.
An evaluator can be reused across fits to retain compiled kernels. The counting
fixture tests best fit, covariance and profile endpoints against Combine's NLL.
For opt-in compiler-cache cleanup, `LocalEvaluator.prepare` lowers and compiles
each term into an explicit executable and returns separate lowering, compilation
and cleanup timings. `release_cache=True` clears process-wide JAX caches, collects
garbage and calls `malloc_trim` where available; retained explicit executables
remain callable. `clear_cache` drops those executables and restores lazy kernels.
`fit(memory_policy="release")` uses this preparation for gradients, drops them
before covariance construction and releases each Hessian executable after adding
its complete contribution. Later profiles recompile gradients as needed. The
default `memory_policy="retain"` keeps the existing fit lifecycle. These options
preserve all curvature entries but do not guarantee a particular peak memory.
Compare full HHVV MX2000 fits, uncertainties and
CMS_B2G20008_scale_j endpoints with Combine, within max(1% of reference, 0.01 sigma).
Measure compilation, setup and steady-state/fit times separately on the same host.
Limit jobs to eight cores and approximately 12 GB RAM; run timing jobs sequentially.

A separate experimental CPU evaluator distributes executable ownership across
spawned workers to avoid one process's mapping limit. Each worker selects a
disjoint set of unchanged local terms; the parent receives individual results
and accumulates them in original global order, adding the model offset once.
Workers set affinity and CPU/float64 configuration before importing JAX.
Aggregate parent/descendant RSS and per-worker mapping guards remain necessary:
partitioning mappings does not itself reduce total memory. Explicit per-gradient
compiler options require independent default-option comparisons followed by full
Combine gradient validation. The fit must use the validated options. The
current spawned evaluator supports gradients only, so its continuation computes
a conditional minimum with covariance explicitly unrequested. A separate complete
fit-point Hessian and covariance comparison remain required.

The alternative bounded CPU fit evaluator keeps all NLL kernels resident and
serializes each gradient executable within its owning process. It validates all
saved points through one bounded load per gradient term before minimization,
then loads the complete gradient term by term only on Minuit gradient requests.
Function-only requests use the resident NLL kernels. Both paths retain global
term order and add the likelihood offset once. Their NLL outputs must agree
within the original tolerance. The optimizer sees canonical NLL minus the
checked initial canonical NLL; gradients are unchanged, and the final canonical
value is evaluated independently and checked against offset restoration.
CPU fusion settings remain explicit and reference-gated. The process has RSS,
mapping and wall guards; keeping gradient lifetimes bounded does not guarantee
that resident values and a single gradient fit within the memory limit. This
minimum-only driver does not supply fit-point curvature or covariance.

A separate continuation consumes an immutable completed minimum after its
kernels have closed. It binds the numerical export, parameter order, categories,
masks, bounds and gradient evidence to that minimum, then computes all
four-column Hessian blocks at the exact fitted vector. NLL terms are independently
reevaluated to check the canonical value. Saved blocks are hash checked and
reassembled before covariance inversion. The complete raw free Hessian is
inverted without regularization or a pseudoinverse, and the original full matrix
is retained including fixed-coordinate rows. Scaled symmetric eigenvalues are
used only for positivity diagnostics; no symmetrized matrix replaces the raw
inverse. Covariance needs positive curvature, small asymmetry and a full inverse
identity check. Minimum and curvature process times are recorded separately
and summed explicitly, excluding the gap between processes. Native covariance
and nuisance-profile comparisons remain required for overall acceptance.

The GPU benchmark separates workspace translation from numerical execution.
Translate with the local Combine v11 / ROOT 6.40 stack and save the trusted
graph/NumPy export with workspace and configuration hashes. On pinned lxplus905,
the LCG 110a CUDA view supplies JAX/jaxlib 0.10.1 and the CUDA plugin; discover
the device and enable float64 before model evaluation. No ROOT is imported in
that process. Native reference fits use an independent node-local v11/ROOT 6.40
venv, after clearing the LCG library and Python environment. Record both stacks.

Benchmark executable storage is a separate, measured policy. The `scoped`
policy loads one trusted executable per local Hessian term and complete-matrix
call, evaluates every column batch, copies each result to the host, then releases
it. Standalone value/gradient call times include loading and cleanup too.
Untimed validation loads each gradient executable once for all 26 reference
points, including a fresh nominal evaluation; that batched throughput is not
reported as standalone latency. GPU Hessians use default compiler options.
The explicitly recorded CPU-only Hessian option disables fusion and sets the
parallel codegen split count to one. Comparisons reject mixed source hashes or
unrecorded policy differences; the different backend compiler options remain
visible in the results.

Save preparation, complete-call samples, every reference output, and the full
nominal Hessian independently of overall run status. Read-only checkpoint
capture must verify matching report/array hashes before preserving a snapshot.
A checkpoint certifies only its completed stage; later failures remain possible.
NVML process memory sampled at 20 Hz supplies the reported GPU peak, with
device-wide baseline/peak retained separately. Supplemental utilization, clock,
temperature and compute-PID samples document shared-device activity without
assigning device-wide utilization to an individual process.

Saved fit and uncertainty comparisons require valid minima and accurate native covariance,
matching workspace/configuration, parameter order, fixed parameters and host.
Check the complete free-parameter Hessian/covariance inverse and compare each
coordinate and uncertainty under GOAL.md's tolerance. Native Minuit errors and
square roots of covariance diagonals are different reported quantities and must
not be substituted silently. Record optimizer/start differences alongside wall
times; these comparisons do not establish a discrete envelope or a profile.
The user's later HHVV speed-comparison policy accepts the four documented native
histogram clipping surfaces and excludes covariance/uncertainty certification
there. That comparison still requires accepted stopping, native same-point NLL,
minimum-NLL and parameter-agreement gates, as specified below.

For the independent profile comparison, freeze a plan after accepted baseline
fits, using one common native starting vector and explicit nuisance values.
Reoptimize the other parameters independently at each point; do not silently
clip values at bounds or label covariance-based grid offsets as profile-likelihood
confidence endpoints. Preserve complete vectors, canonical NLLs, POI shifts and
optimizer validity for every point. Check native zero-point restoration and
evaluation errors, then compare NLLs, all free coordinates and POI shifts.
Publish profile arrays via a temporary file; update the array and report while
holding the same lock used by the resource observer, preserving the preceding
complete artifact if a write is interrupted. Profile elapsed times include
device initialization and model/plan validation after CLI checks, with Python
startup and earlier module imports explicitly excluded.

The bounded CPU profile policy prepares resident NLL kernels and disk-backed
gradients once for the whole grid. It requires an accepted baseline plan and
source-bound completed bounded minimum. All 26 fresh NLL and gradient arrays
must reproduce that minimum's accepted reference batch bitwise; otherwise
profiling stops pending a new reference comparison. A separate optimizer
parameter view holds the chosen nuisance at each requested value, preserving
the original likelihood and shared evaluators. Each point starts from the same
native baseline vector with only that coordinate changed. The constant-centered
objective restores canonical NLLs, and the full derivative is supplied to
Minuit, which ignores fixed coordinates. Source/reference integrity, mappings,
RSS and wall limits are checked. The final profile report is published after
the evaluator has closed, and elapsed time includes that cleanup.


The optional `torch_backend` adapter consumes the same JSON/NPZ and runtime buffers.
Counting, stacked constraints and CMS templates/profile solves share the neutral
operator equations. Indexed updates are functional so autograd and repeated calls
preserve inputs; reductions stay on the selected device. Indices convert to int64
on transfer, and floating arrays/coordinates remain float64. Binned histograms,
mixtures and coordinate programs use the same neutral schedules. Analytic special
functions have a separate PyTorch port that preserves native branches, leaving
the certified JAX operator source unchanged. Full HHVV CPU values/gradients and
generic-card Hessians pass; CUDA-graph capture and performance are separate
acceptance steps. The local CPU package is
2.11.0+cpu, matching the LCG PyTorch 2.11 release without importing its CUDA stack.
Products use a balanced multiplication tree: PyTorch 2.11's built-in product
backward reads a zero count on the host and fails CUDA capture. The tree retains
polynomial first and higher derivatives at zero and signed factors, with no
data-dependent host reads. CPU derivative tests and changed-input T4 graph replay
cover this replacement; full HHVV CPU and generic T4 native gates pass afterward.

## Validation gates

The tensor backend uses a separate device fit path. Framework-neutral parameter
metadata preserves bounds, roles and errors alongside the model. An explicit
`from_workspace(..., fit_policy='combine_v11_default')` import applies v11's
`OPTIMIZE_BOUNDS=1, UNBOUND_GAUSSIANS=0` policy before extraction: marked
ModelConfig nuisances lose both limits. This ordering also accounts for bounds
used in constraint normalization. The input workspace ranges are restored on
exit; the neutral artifact keeps stored and effective bounds separately, plus
fixed values and non-observable category choices. The default `workspace` mode
retains the supplied ranges for existing oracle comparisons.

Native fit timing requires an explicit policy artifact. The CLI receives the
imported `MH` through `--mass`; its default 120 must not silently replace a
workspace value of 125. Before fitting, the harness runs the actual CLI with
`--skipInitialFit`, checks the `clean` starting snapshot and the effective
`MultiDimFit` snapshot, and records the check separately from fit wall time.
Same-point native checks apply the identical effective ranges. A missing or
mismatched state prevents agreement claims. No histograms are repaired.

Host setup removes equal-bound and explicitly held coordinates, then scales the free
coordinates using native errors (or a documented optimization-only fallback).
The numeric model, coordinate map, scales and bounds are runtime arguments.
Both scaled limits and exact physical limits are carried. A fused affine map can
round just outside a zero physical bound even at a feasible scaled endpoint.
The physical value is corrected with those exact limits; a stopped-gradient
roundoff correction preserves the affine AD derivative and raw external Hessian.
This does not clip the mathematical likelihood or change its convergence target.
The coordinate regression checks the known minimum and raw covariance, including
a unit derivative with respect to the physical coordinate at an active bound.
BFGS iterations and feasible Armijo backtracking use nested `lax.while_loop`s;
parameters stay on device until the complete fit returns. By default, convergence requires
the infinity norm of the box-projected gradient to meet the requested tolerance.
Iteration limits, failed searches and nonfinite values are explicit statuses.
The same active-bound mask is used for projected gradients, search directions
and the BFGS secant update. A blocked coordinate's row and column are decoupled
from the search metric, with a unit diagonal so it can move if its gradient
later points into the feasible region. Its gradient change is excluded from the
free-coordinate curvature update. Retaining that change while only projecting
the search direction caused cycling on the coupled convex quadratic regression
set. Completed HHVV strict-gradient diagnostics still fail at their iteration
limit; their saved native NLL checks do not certify convergence.
Steps limited by a bound land on that exact bound, avoiding a one-ulp interior
point that the active mask would otherwise freeze. These changes affect the
search metric only; the final covariance still comes from exact AD curvature.
Armijo comparisons include 8×machine-epsilon×(1+abs(NLL)) to permit decreases
unresolved in the rounded objective; zero-displacement trials are rejected.
Every use of this allowance is counted, and the maximum allowance is returned.
The projected-gradient convergence threshold is never relaxed. Setting the
multiplier to zero retains strict Armijo comparison for diagnostics.

The same outer JIT evaluates the exact AD Hessian at convergence and inverts the
raw free matrix. The BFGS approximation supplies search directions and, when
requested, estimated EDM; it is never the reported covariance. Finite
curvature, small asymmetry, a positive Cholesky factor and a small inverse identity
residual are required for covariance acceptance. Failure produces a status and
NaN covariance, without repair. Scales transform the inverse to physical units;
held coordinates have zero rows/columns. Active-bound covariance is local
curvature, not a bounded confidence interval. Stationarity alone is not accepted
as positive curvature. Small-card native-objective comparisons are recorded in
`reports/tensor_device_fits.json`; full-fit CPU/GPU benchmarks remain separate.
The benchmark harness saves an immutable first-fit array checkpoint before warm
repetitions. A failed fit or invalid covariance skips those repetitions and is
reported as a failed attempt; checkpoint transfer and I/O are outside execution
timings. This does not change optimizer or covariance acceptance criteria.

An explicit `stopping_criterion='edm'` supports the user's accepted speed
comparison. Combine v11 defaults are Minuit2/Migrad, strategy 1, tolerance 0.1.
Its NLL has error definition Up=0.5: the normalized stopping condition
EDM/Up < 0.002×tolerance corresponds to raw NLL EDM < **1e-4**. Source, CLI and
installed ROOT probes are saved in `reports/tensor_combine_stopping_rule.json`.
The strict projected-gradient mode and its 1e-7 HHVV diagnostics remain separate.

Current HHVV comparisons use a single saved Combine-standard pre-fit Asimov:
`GenerateOnly -t -1`, nominal workspace nuisances/POIs, nominal `r_hhvv`, MH=125,
and frozen MX=2000. Generation does not use a fitted snapshot or
`--toysFrequentist`. The importer selects `asimov_prefit`; native fits use
`-D asimov_prefit`. Dataset identity includes every channel, coordinate and
weight, including zero weights. The generation report, workspace, imported
buffers and reference arrays are bound by hashes before timing or fit output
access. The native v11/ROOT 6.40 process stays isolated from the CUDA LCG view.
Historical observed-data fit values are excluded from new work; their existing
reports are identified by filename without reproducing values or rewriting history.

Native derivative references use the fixed nominal + 25 native-valid random
points. Independent workers can capture disjoint subsets; the audit binds each
coordinate and canonical NLL to the original point archive and rejects duplicate
assignments. For subtraction-sensitive derivatives, a separate native-only
diagnostic reads live Combine cached components, exact native integration modes
and constraints. It forms long-double NLL differences with stable logarithmic
remainders and per-bin Poisson reductions. Each difference is checked against
the native NLL. A fixed step grid and convergence rule select references before
AD comparison; the NLL and gradient acceptance tolerances remain unchanged.
Incomplete references cannot establish a fit-speed claim.
A native shard can hand off its remaining indices after an identified worker
stops at a fully saved prefix. The sealed checkpoint binds the exact original
report bytes and retains its original planned indices. Its separate ownership
list covers only saved points; the merger rejects overlaps, missing prefix
points and unverified termination. This changes scheduling, not reference
sampling, numerical evaluation or acceptance.


A separate diagnostic linearizes the native NLL at the reference point before
finite differencing native histogram densities. For a complete, unconditionally
normalized histogram, `sum(volume * dp) = 0`. Its density adjoint can therefore
be centered by subtracting `lambda * volume` without changing the derivative.
This suppresses normalization roundoff near an Asimov point. The helper verifies
the native histogram class, full grid, original ranges and unit normalization;
conditional histograms, floating mixture coefficients and other affected shapes
are explicit unsupported cases. The first-order function differs from NLL
changes at finite steps, so the full native NLL reconstruction is checked
separately on every probe. All histogram-score references are confirmed using
the first six consecutive valid, distinct descending steps with the same
stencil/direction and full relative spread below 1e-5. This confirms a first
three-step plateau with a second three-step window; one original short plateau
moved at smaller steps. The full fixed 19-step native grid is retained, selection
uses no AD values, and all six selected estimates must satisfy the unchanged
1e-4 relative AD gate. Original three-probe comparisons remain in the report.

Cache-delta references also prefer the first converged six-step window when
one exists. This native-only rule is applied to every cache-delta case before
reading AD values; otherwise the existing three-step policy remains in force.
The audit records the six-step search outcome and the original comparison in
both cases. A point-20 `jmrAK816` scan demonstrated why: a coarse three-step
plateau moved at smaller steps, while six smaller steps agreed with one another.
The original failed AD comparison is retained. Neither finite-difference spread
nor AD relative tolerance changes; histogram-score references still require six
steps and cannot fall back to three.
This is a reference diagnostic, with no likelihood or optimizer modification.


Device BFGS estimates EDM as half the projected-gradient quadratic form in its
inverse search metric, corrected by `(1+3*inverse_uncertainty)`. The uncertainty
starts at one and is updated from the relative matrix change. A nonstationary
fit needs a secant update before this estimate can stop it. Smooth EDM fits
default to `edm_validation='exact_hessian'`: valid positive exact curvature
checks the same EDM threshold, and an unmet threshold restarts BFGS on device
with that inverse metric. Up to three refinements share the original iteration
budget. The EDM metric fixes bound-blocked directions; the reported covariance
still uses the full unmodified free Hessian. Invalid curvature remains explicit,
and exhausted refinement yields `edm_validation_failed`. The initial autoMCStats
premature-stop case and passing refined comparisons are retained in the reports.

HHVV explicitly selects `edm_validation='estimated'` under the accepted clipping
policy. Exact final curvature remains a diagnostic without triggering refinement
or certifying uncertainties. Successful native agreement requires a 1e-6
same-point NLL check, minimum NLL difference at most the sum of the two stopping
budgets (2e-4), and parameter differences within max(1% of the native value,
0.01 native sigma). Every parameter contributes to the reported maximum
|delta theta|/sigma, with r_hhvv and leading nuisances listed. The native sigma
is a comparison scale at HHVV's nonsmooth point, not a certified uncertainty.

Native timing uses `MultiDimFit --algo none --saveWorkspace`: requesting
`--saveFitResult` would also request post-fit HESSE/MINOS. Strategy 1 retains its
own internal curvature operations. Forwarding wrappers count CachingSimNLL
evaluations and numerical-gradient invocations, suppressing nested forwarding
counts; their fitted values, errors, NLL, EDM and Nfcn are bit-identical to the
uninstrumented counting/autoMCStats runs. Native internal minimization time and
complete executable wall time are separate. Tensor lowering, compilation and
synchronized first-fit execution are separate; warm repeats start from the same
nominal vector. Reports distinguish optimizer NLL/gradient calls from final AD
Hessian linearizations. A successful-fit speed ratio requires the agreement
gates and the completed same-node timing evidence; failed-attempt times are
never presented as successful-fit speedups.

Fit results expose two explicit error conventions. `errors` and
`curvature_errors` are sqrt(diagonal(raw inverse AD Hessian)). `reported_errors`
maps these through the finite internal displacements used by
[Minuit's error-reporting transformation](https://root.cern.ch/doc/backup_latest-stable/MnUserTransformation_8cxx_source.html).
This distinction matters near bounds; reported errors are not profile intervals.
The mapping runs in the final device computation and never changes the raw
covariance. A per-coordinate status marks fixed parameters, invalid covariance
and numerical boundaries where external covariance cannot reliably reconstruct
the internal step. The implementation is checked against the installed ROOT
transformation and actual native fits; these two quantities are never silently
substituted in validation.

`tensor.profiling.prepare_profile_scan` validates a finite physical grid and
constructs runtime coordinate maps with the requested parameter held fixed.
One outer JIT maps the bounded fit sequentially over those coordinates using
`lax.map`; all inner BFGS and line-search loops remain on device. Each point uses
the same supplied nuisance start. Live optimization storage is one BFGS workspace,
and output contains point values, fitted vectors, projected gradients and explicit
statuses. Final covariance matrix outputs are omitted from scans. The default
strict-gradient scan omits curvature calculations; an explicitly requested
exact-Hessian EDM check can compute curvature internally for stopping validation.
Each point retains estimated and corrected EDM, exact-Hessian EDM, its validation
flag, refinement count and Hessian evaluation count. Explicit estimated-EDM scans
skip curvature and report the exact-Hessian EDM as unavailable; their status does
not certify curvature at a clipping surface. These diagnostics distinguish the
accepted EDM rule from the separate strict-gradient criterion.
Failed points remain failures and cannot silently participate in an envelope.
Discrete branches are distinct frozen models; the generic common-grid validation
includes the native correction and a change of preferred branch. Grid validation
does not establish confidence endpoints or the HHVV nuisance scan.

The native benchmark accepts repeated `--fixed-parameter NAME VALUE` arguments
for a profile point. Its small neutral fit-state view shares the large likelihood
buffers, fixes scalar input slots and retains the original dataset provenance.
At the ROOT boundary it mirrors Combine's freeze operation, including removal
from the nuisance sets of `ModelConfig` and `ModelConfig_bonly`. The actual CLI
preflight checks this state before minimization; native postchecks retain the
canonical NLL. The benchmark's existing HHVV Asimov guard runs before profile
configuration. This entry point currently requires coordinates inside the stored
workspace ranges and rejects other values rather than clipping them. Counting
and autoMCStats profile fits are checked against the device EDM scan, including
parameter, minimum-NLL and same-point NLL gates. HHVV profiling remains separate
work, with its accepted clipping caveat unchanged.

`scripts/benchmark_tensor_profile.py` captures one complete device scan in a
ROOT-free process. It verifies the device before model work, applies the HHVV
Asimov guard, separates lowering/compilation from synchronized execution, and
saves every point's fitted vector, EDM diagnostics and evaluation counts in NPZ.
It uses the same persistent structure cache and static allocation/RSS guards as
the single-fit benchmark. A failed point stays in the capture and cannot imply
profile agreement. CPU integration tests cover multiple grid points, native NLL
postchecks, cache writes and rejection of unblinded HHVV metadata before fitting.

`scripts/tensor_profile_plan.py` recomputes the accepted native/device baseline
comparison before deriving physical nuisance points from native fitted values
and errors. The plan binds the model, workspace, data, fit state, producer sources
and baseline reports. It records one imported nominal nuisance start for every
point. Native errors define grid spacing; they do not establish confidence
endpoints. Loading a plan repeats its baseline and checksum checks, and values
outside native stored ranges are rejected without clipping.

The device benchmark accepts `--plan`; the native benchmark accepts
`--profile-plan` with `--profile-index`. These options supply the physical grid
and cannot be combined with independently specified fixed values. The saved
point checker verifies native fixed-coordinate state and canonical/offset NLL
restoration. `audit_tensor_profile_comparison.py` requires every grid point to
pass both stopping rules, the unchanged same-point NLL gate, the 2e-4 combined
minimum-NLL budget, and all coordinate gates. A shared plan additionally compares
POI shifts against its accepted native baseline, with tolerance
`max(1% of the native shift, 0.01 native baseline sigma)`. This prevents a small
shift error being hidden by a large absolute POI value. Missing or failed points
cannot form an accepted profile, and this audit makes no scan speedup claim.

Persistent executable storage uses an explicit trusted cache directory, partitioned
by the neutral model's structure key. JAX's own cache key validates the lowered
program, arguments, compilation settings and device/version compatibility within
that namespace. Numeric buffer changes do not change the model structure key.
Configure this before JIT dispatch and outside concurrent compilation; changing
the configured model resets JAX's disk-cache handle without deleting entries.
Auxiliary XLA caches are disabled by this API. LCG JAX 0.10.1 includes the
absolute per-fusion autotuning-cache path in its executable key, so moving job
scratch otherwise causes a miss. Disabling the optional auxiliary cache leaves
JAX's executable compatibility checks unchanged. Relocation tests under both
JAX 0.10.1 and 0.12.0 require NLL/fit cache hits and unchanged executable bytes,
while runtime data and fit results change. Full-HHVV NLL and gradient executables
also pass [transfer between two A100 Condor workers](reports/gpu_tensor_portable_cache_cross_job_summary.json):
unchanged keys and executable bytes, explicit cache hits, and all 26 native
validation points. The measured second-job loads take 0.756/2.155 s; lowering
still takes 5.480/14.120 s.
Fresh-process tests require explicit NLL/fit cache-hit logs, unchanged executable
checksums, and correct changed-data outputs. This does not eliminate Python
tracing/lowering, establish compile-time scaling or prove cross-hardware reuse.

Each class needs ROOT/Combine value and integral comparisons over its relevant
branches and AD/finite-difference checks. Generic counting, autoMCStats template,
parametric mixture and RooMultiPdf cards precede HHVV per-channel and global tests.
Compare nominal and seeded random points within +/-2 sigma. Required absolute NLL
error is below 1e-6; gradient relative error below 1e-4 (report absolute error too,
and explicitly handle exactly zero reference derivatives). Verify Hessians with
finite differences of gradients. Keep reproducible scripts, complete point vectors,
versions, options and comparison results. Old v10 prototype logs do not certify v11.

Audit every supplied workspace for v11/ROOT 6.40 readability. Rebuild from the
read-only original cards only if required. Independently probe ROOT 6.40 codegen/Clad
on a supported toy and the same HHVV workspace; record failures as coverage gaps,
not timings. Completion requires all GOAL.md gates, not merely registry coverage.

### Cancelled MX3000 observed limit (2026-10-28; historical design)

The user revoked the combined MX3000 unblinding exception and cancelled this
work: **observed cancelled, no result; HHVV is blind at every mass again**.
Do not resubmit observed jobs. The following describes the historical
implementation, not current authorization. The dedicated worker verifies the
workspace/card hashes, all 459 channels, MX, MH=125 and the three fixed nominal
RooMultiPdf choices before importing data_obs.
Existing general CLI blinding guards remain in place.

Observed CLs uses two profile likelihoods: real data and the background-only
Asimov conditioned on the r_hhvv=0 fit to that real data. This is Combine's
`--run observed` default, not `--noFitAsimov`. JAX supplies that fit; a separate
native v10 / ROOT 6.30 process restores analytically profiled BB-lite parameters,
checks the native same-point NLL, calls `asimovDatasetWithFit(needsFit=false)`,
and saves its constraint/global-observable snapshot. Import takes place separately
in pip v11 / ROOT 6.40 with explicit `v10_density`. ROOT versions are never mixed
in one process. Both global fits have nonnegative POI bounds and Up=0.5; global
and conditional fits require corrected estimated raw-NLL EDM <1e-4.

`observed_limit` implements the two branches of Combine's q-tilde CLs formula
and its bisection/log-interpolation search, CL=0.95, rAbsAccuracy=0.0005,
rRelAccuracy=0.005 and up to five upper-bracket expansions. The GPU driver uses
the native 2%-of-rMax error floor to seed bracketing, without a Hesse uncertainty;
this can alter the final point within the documented stopping accuracy. It does
not claim a native-identical optimization trajectory. Failed fits, absent upper
brackets and nonfinite statistics are errors, never accepted observed limits.
The expected-only implementation and previous validation reports stay intact.

## RooMultiPdf profiling: staged implementation

The default importer still freezes the selected alternative, preserving the old
backend as an oracle. Opt-in tensor import with `discrete_policy="profile"` now
retains the union of active alternatives' parameters, normalized density programs,
and stored correction factors. Category indices are integer runtime buffers;
changing them does not retrace the likelihood. A fixed category keeps the old
selection. Constant templates are still folded, but their categorical selection
stays dynamic. Penalties enter the raw NLL once per channel in the same flattened
RooAddPdf locations as CachingAddNLL; they are not recomputed from the number of
parameters left floating after freezing. The public limits CLI defaults to
rejecting discrete models; `--discrete-policy profile` explicitly selects the
new implementation, whose method-level native validation is reported separately.

Native v10's default CascadeMinimizer performs an initial continuous fit, then
alternates short scans over orthogonal index combinations with full continuous
fits, stopping at a change below 0.001 NLL or after 15 iterations. During each
short scan it freezes POIs and parameters outside the RooMultiPdf parameter
union. Dhi additionally freezes inactive alternative parameters, hides constants,
masks constraints, and uses `MINIMIZER_multiMin_maskChannels=2`. Dhi's default
Migrad strategy is 0 with tolerance 0.1 and fallbacks at 0.2 and 0.4. These
conventions require their own native comparisons; a frozen-choice limit is not
a discrete-profile result.

The tensor implementation uses a device `while_loop` around full fits and a
`scan` over cyclically reordered orthogonal combinations. The short objective
contains only the compact discrete-channel program and freezes all common
parameters. A change of category selects a new parameter mask without changing
the compiled coordinate layout. Candidate fits reset the initially active
parameters, as native v10 does; inactive alternatives retain their stored state.
Every candidate must converge, as must the final full fit and the 0.001-NLL
discrete iteration. Candidate status/EDM/counts are retained. The 15-iteration
cap cannot silently turn an unresolved envelope into a successful result.

This first implementation explicitly rejects constrained discrete shape
parameters, shape parameters shared between categories or with non-discrete
channels, and nested MultiPdfs. Those cases need additional native masking-offset
validation. Common signal/yield parameters can still be shared and float in the
full fit. These restrictions make differences of the compact channel NLLs an
exact ranking of the full candidates while common parameters are frozen.

Nominal errors attached to inactive workspace alternatives can be tiny despite
appreciable allowed shape changes. For discrete shape coordinates only, the
optimization scale is floored at 1% of a finite native parameter interval, or
0.1*max(abs(nominal),1) for an unbounded interval. This prevents an uninformative
initial BFGS metric from preventing all secant updates. It does not change
starting values, physical bounds, float64 precision, EDM acceptance, penalties,
or reported uncertainties. The fitter remains BFGS with the native stopping
threshold, not a claimed identical Minuit2 trajectory or fallback implementation.

The expected-limit profiler constructs the full and compact discrete-channel
objectives once and shares those immutable closures between global and
conditional solvers. This avoids repeated host-side array validation; model
buffers, numerical kernels and stopping rules are unchanged. After the global
fit, its parameter vector is exported once into the receipt for explicit
crossing diagnostics. Conditional fits retain their device-resident warm starts.

If an expected crossing is unresolved, the driver additionally profiles at the
fitted POI and at zero, each independently seeded from the saved global nuisance
vector and discrete choices. It records the seed NLL and conditional result,
then restores the usual warm-start state. These diagnostics reuse the compiled
solvers and unchanged stopping rules. They never replace the reported baseline,
crossing trace or limit; failed fits remain explicit and an unresolved crossing
remains unavailable. The additional calls count in execution time and fit totals.

The runtime passes temporary neutral-model ownership into the expected-limit
driver. It prepares device inputs/objective closures, releases the host model,
then compiles. The profiler also supports immediate compilation for existing
callers; a caller retaining its model retains that memory. Static allocation
and aggregate RSS guards are unchanged. RSS failures retain process-level
measurements and diagnostic PSS/private-page counts; PSS does not replace the
RSS acceptance rule.

Neutral-model archive loading streams the checksum and retains each freshly
decompressed numeric array as immutable, avoiding a second complete host copy.
The public bundle constructor still copies caller-owned arrays. Both paths
enforce the same numeric schema, C-contiguous layout and structure checksum;
no template precision or model contents change.
