# combine-jax

> This public repository holds the code, scripts and tests only. The validation reports and `STATUS.md` that this README links to are kept in the development history and are not published here.

A backend under development for CMS Combine v11 likelihoods. The tensor path
imports a workspace and datacard into framework-neutral arrays, then evaluates
one complete batched float64 likelihood. It includes device fits and profile
scans; the retained registry-based graph backend serves as the reference oracle.
Unsupported classes and configurations raise named errors.

Generic-card fits and profiles pass native-objective checks. Current HHVV MX2000
benchmarks use only the same saved Combine nominal pre-fit Asimov dataset in both
backends. Local CPU, lxplus905 CPU, A100 CPU/GPU and H100 CPU/GPU pass the full
native NLL and gradient gates at nominal + 25 random points. T4 NLL checks pass;
converged-fit comparisons and the requested HHVV nuisance impact remain open.
HHVV covariance at the accepted clipping surfaces is uncertified.
Earlier full-channel CPU/T4/A100/H100 validation and Hessian comparisons used the
historical dataset and establish no Asimov fit speedup. See [GOAL.md](GOAL.md)
for acceptance criteria and [STATUS.md](STATUS.md) for exact validation scope,
measured timings, environments and remaining gaps.

Use the project environment and resource settings:

```bash
source .venv/bin/activate
source scripts/env.sh
pip install -e . --no-deps --no-build-isolation
python scripts/validate_counting.py
python scripts/validate_templates.py
python scripts/validate_parametric.py
python scripts/validate_multipdf.py
python scripts/validate_hhvv_histfunc.py
python scripts/validate_hhvv_errors.py
python scripts/validate_hhvv_vertical.py
python scripts/validate_hhvv_vertical_nd.py
python scripts/validate_hhvv_splines.py
python scripts/validate_hhvv_products.py
python scripts/validate_hhvv_resonance.py
python scripts/validate_hhvv_remaining_shapes.py
python scripts/validate_hhvv_nominal.py
combine-jax inspect build/counting.root --output build/counting_inspection.json
```

`scripts/env.sh` limits CPU affinity to eight available cores and BLAS/OpenMP to one
thread, selects CPU JAX/float64, and keeps temporary files in `build/tmp`.
The ROOT loader locates the installed Combine wheel's private headers and library;
no CMSSW shell environment is needed.

For the complete tensor test selection, use the guarded module runner on a clean
source worktree. It runs each module in a fresh process with two logical CPUs, a
4 GiB process-tree RSS limit and a 900-second limit per module. The project venv
supplies dependencies; `PYTHONPATH` selects the worktree's source and scripts.

```bash
git worktree add --detach build/tensor-tests HEAD
python scripts/check_tensor_release_suite.py \
  --worktree build/tensor-tests --output reports/tensor-tests.json
```

Use a new output path for each run. The JSON report records source hashes, module
exit codes and log hashes. `--resume <previous-report>` reuses only passed modules
whose source and log hashes still match. The last audited selection contains
[218 passing tests across 21 modules](reports/tensor_release_suite_audit.json).
Run the remaining tests with `pytest -q --ignore-glob='tests/test_tensor_*.py'`,
or select individual pytest files for a focused change.

## Tensor import and execution

Import once with the verified Combine v11 / ROOT 6.40 environment. The card and
workspace must describe the same channels. This example uses a counting card;
the importer selects supported operator families from the workspace itself.

```python
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace

with open_workspace("build/counting.root") as workspace:
    model = from_workspace(workspace, "tests/cards/counting.txt",
                           fit_policy="combine_v11_default")
model.save("build/counting_tensor")
```

This policy applies Combine's effective nuisance ranges before import and keeps
the workspace's mass. The default `fit_policy="workspace"` retains stored ranges
for oracle comparisons. Native CLI comparisons also pass the imported mass
explicitly; Combine's CLI mass default is 120.

For HHVV, `scripts/generate_hhvv_prefit_asimov.py` runs Combine `GenerateOnly -t -1`
at nominal nuisances/POIs, nominal `r_hhvv` and MH=125. Then
`scripts/import_hhvv_prefit_asimov.py` imports the saved `asimov_prefit` dataset,
verifies its complete row/weight identity, and records generation provenance.
The underlying named-data API is `from_workspace(..., data="asimov_prefit",
fit_policy="combine_v11_default")`. Benchmark fits use the verified import and
native `-D asimov_prefit`; no observed-data HHVV fit is part of the current campaign.

The saved `model.json` and numeric `arrays.npz` contain no ROOT/JAX/PyTorch objects.
Parameter-independent quantities are precomputed at import. A numeric worker can
load these files without ROOT, including in the CUDA LCG view. Select its JAX
platform before importing JAX, enable float64, and verify the selected devices.
Do not source a different ROOT stack into the importer environment.
For CUDA, set `JAX_PLATFORMS=cuda` in the numeric worker's environment before
starting Python. The supplied `scripts/benchmark_tensor.py --backend gpu` worker
requires an actual GPU and float64 before loading model arrays.

```python
import jax
jax.config.update("jax_enable_x64", True)
print(jax.devices())  # Verify a GPU here when requesting CUDA execution.

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.tensor.compilation import configure_model_cache

model = TensorBundle.load("build/counting_tensor")
configure_model_cache(model, "build/jax_cache")  # Optional trusted cache directory.
info = model_parameter_info(model)
theta = jax.device_put(info.arrays["initial"])
arrays = jax.device_put({k: v for k, v in model.arrays.items()
                        if not k.startswith("parameter_info__")})
value_and_gradient = jax.jit(jax.value_and_grad(make_model_function(model)))
value, gradient = value_and_gradient(theta, arrays)
jax.block_until_ready((value, gradient))
```

Model buffers are runtime arguments, so changing data does not embed a new copy in
the executable. Cache namespaces use model structure; JAX additionally checks
executable compatibility. Cache reuse still requires tracing/lowering. Exact
fresh-process and batch-job reuse evidence is tracked in `STATUS.md`.

For CERN Condor jobs staged under the pinned node's `/tmp/sqian/combine-jax`,
submit with `condor_submit -spool job.sub` and retrieve completed outputs with
`condor_transfer_data <job-id>`. Spooling copies the inputs to Condor; an ordinary
submission cannot make node-local `/tmp` files visible to the remote scheduler.
Keep payloads off AFS and verify the assigned GPU before numerical work.

```python
from combine_jax.tensor.fitting import prepare_fit
from combine_jax.tensor.profiling import prepare_profile_scan

fit_options = dict(stopping_criterion="edm", edm_tolerance=1e-4,
                   edm_validation="exact_hessian")
fit_function, coordinates, fit_arrays = prepare_fit(model, **fit_options)
fit = jax.device_get(fit_function(coordinates, fit_arrays))
if (fit["status"] != 0 or not fit["valid_covariance"]
        or not fit["edm_curvature_validated"]):
    raise RuntimeError("Fit or final covariance did not pass its validity checks")

scan, scan_coordinates, scan_arrays = prepare_profile_scan(
    model, "r", [0., .5, 1., 1.5, 2.], initial=fit["values"], **fit_options)
profile = jax.device_get(scan(scan_coordinates, scan_arrays))
if (profile["status"] != 0).any() or not profile["edm_curvature_validated"].all():
    raise RuntimeError("At least one profile fit did not converge")
delta_nll = profile["value"] - fit["value"]
```

This generic-card example uses the raw NLL EDM target corresponding to Combine
v11 Minuit2/Migrad, strategy 1, tolerance 0.1 and Up=0.5. The tensor optimizer
uses BFGS and checks exact final curvature for smooth objectives. The HHVV
campaign explicitly selects `edm_validation="estimated"` under the accepted
clipping policy, records the final exact Hessian as a diagnostic, and requires
independent native minimum agreement. Its uncertainties remain uncertified.
The separate `stopping_criterion="projected_gradient"` mode retains a configurable
`gradient_tolerance`; it is not the HHVV speed-comparison criterion.

BFGS iterations and line searches stay on the device. The fit returns the exact
final AD Hessian and its raw inverse; invalid curvature is reported, never
repaired. `errors` are curvature errors; `reported_errors` separately reproduces
Minuit's bounded reporting convention. Neither is a profile interval.

The complete scan uses a sequential device loop and omits matrix outputs. Exact
EDM validation can compute Hessians internally; estimated-EDM and strict-gradient
scans skip that curvature work. Its executable accepts new grids of the same length and parameter layout
through runtime coordinates. Every point starts from the supplied nuisance
reference. A scan does not prove a global minimum or determine confidence
endpoints automatically. RooMultiPdf alternatives remain separate frozen models;
compare their converged profiles, including the imported correction penalties.

For native/device profile comparisons, `scripts/tensor_profile_plan.py` derives
one shared physical grid from an accepted baseline fit comparison. It rechecks
the baseline gates and binds the model, dataset, sources and reports. Use this
plan with `benchmark_tensor_profile.py --plan` and with
`benchmark_combine_fit.py --profile-plan --profile-index`; both runners use the
same recorded grid and nominal nuisance starts. The native point checker accepts
`--profile-index` for saved scan outputs. `audit_tensor_profile_comparison.py`
checks every native/device minimum and the POI shifts against the shared
baseline. Counting and autoMCStats grid comparisons, and a baseline-linked
counting POI-shift comparison, pass; the HHVV scan is still pending. Native
errors used for grid spacing do not establish confidence endpoints.

An optional `combine_jax.tensor.torch_backend` adapter consumes the same artifact
via `make_torch_function` and `to_torch_arrays`. It supports float64 eager execution
and the CUDA-graph benchmark harness. It does not currently provide a Torch fit
driver; accepted numerical/performance scopes are listed in `STATUS.md`.

## Retained graph API and diagnostic tools

```python
from combine_jax.root_io import open_workspace
from combine_jax.likelihood import from_workspace

with open_workspace("build/counting.root") as workspace:
    likelihood = from_workspace(workspace)

print(likelihood.parameter_names)
nll, gradient = likelihood.value_and_grad(likelihood.initial)
hessian = likelihood.hessian(likelihood.initial)

# Compile terms with local parameter slices, then assemble NumPy derivatives.
local = likelihood.local_evaluator()
nll, gradient = local.value_and_grad(local.initial)
hessian = local.hessian(local.initial, retain_compiled=False)

from combine_jax.fitting import fit, profile
result = fit(likelihood, evaluator=local)
print(result.valid_minimum, result.values, result.errors)
# Profile any active continuous parameter at specified physical values.
baseline, points = profile(likelihood, "r", [0.5, 1.0, 1.5], best_fit=result,
                           evaluator=local, covariance=False)
```

Exported likelihoods contain no live ROOT objects and remain usable after the file
closes. The canonical objective clears Combine's constant and explicit zero points;
`likelihood.at_zero(theta)` creates a separately offset objective. Actual data
coordinates, continuous PDF normalisation and extended process yields are preserved.
For supported autoMCStats channels, total Gaussian bin nuisances are profiled by
default. `likelihood.profile_values(theta)` returns their values; pass
`analytic_barlow_beeston=False` to `from_workspace` to retain them as explicit
parameters. Profiled solutions outside parameter bounds or with zero total error
are marked NaN; zero error otherwise leaves Combine's mutable previous value.
Class translators live in `src/combine_jax/`; the architecture and
treatment of constraints, discrete choices and bin statistics are in
[DESIGN.md](DESIGN.md).

RooMultiPdf selections and external channel masks are frozen from workspace state
at export; re-export after changing them. Inactive alternatives do not introduce
unused AD parameters. Masked channel constraints remain in the likelihood.
The local evaluator groups profiled-bin constraints with their channel, compiles
only each term's active parameters, and adds gradient/Hessian blocks in the original
parameter order. Its host API returns NumPy arrays; the original likelihood API
remains directly transformable with JAX. Full HHVV derivative scaling is still being
validated.

The fit driver supplies JAX gradients to iminuit and obtains covariance by
inverting the free-parameter block of the autodiff Hessian. It reports nonpositive
curvature, fixed coordinates, parameter bounds and Minuit validity separately.
Generic counting-model fits and profiles are tested against Combine's NLL; HHVV
fit and uncertainty comparisons remain open.
`scripts/compare_exported_fit.py` compares saved JAX and native fit reports,
verifies their array hashes and common model/host, and checks the GOAL.md
coordinate and uncertainty tolerances. It keeps native Minuit-reported errors
and covariance-diagonal sigmas separate. An incomplete fit cannot pass this gate;
whole-run times are reported with differences in optimizer settings and starts.
`fit_combine_reference.py --hesse-strategy 1` selects the quality of explicit
native HESSE separately from MIGRAD. Reports record the actual HESSE settings;
the same minimum and covariance acceptance gates still apply.
For a reproducible native/JAX nuisance profile, `scripts/profile_plan.py` freezes
common physical points from accepted baseline fits. `scripts/run_profile_points.py`
reoptimizes the other parameters using either backend, and
`scripts/compare_profile_points.py` checks every saved point and POI shift.
Native and bounded CPU profiles accept `--migrad-iterate 1 --no-simplex`
to request a single MIGRAD pass without simplex fallback. Defaults remain five
passes with fallback. Nondefault retry controls require `--engine combine` or
`--engine cpu --bounded-minimum ...`; other modes reject them explicitly.
The comparison requires the same recorded optimizer settings for both engines.
Grid offsets use native Minuit errors; they are not automatically confidence
interval endpoints. The long-run wrappers preserve partial results and prevent
overlap with an active lxplus benchmark. The counting workflow passes locally;
the full HHVV profile and GPU profile runtime remain unvalidated.

For local CPU memory experiments, `scripts/sharded_cpu_evaluator.py` assigns
retained gradient kernels to separately spawned workers and sums every term in
the original order. It checks aggregate RSS and per-worker mappings. The
measurement runner exposes an explicit `--cpu-no-fusion` option, gated by saved
compiler comparisons; lower memory can come with slower calls. The default
full HHVV trial and its no-fusion retry both hit the aggregate-memory guard;
their terminal evidence is recorded in STATUS.md. `scripts/run_sharded_fit_after_validation.py` queues a
minimum-only fit only after complete, source-bound Combine gradient checks;
it does not supply a covariance or establish full fit acceptance.

`scripts/fit_bounded_cpu.py` tests a second local CPU policy: retain the smaller
NLL kernels, but load gradient kernels one at a time only when Minuit requests
a gradient. Before fitting, it checks every saved reference point and requires
the frozen HHVV component/directional comparisons. It centers Minuit's objective
at the checked starting NLL and restores canonical values in its reports.
RSS, mappings and wall time have explicit guards. This experimental driver
produces a conditional minimum; a complete fit-point AD covariance is still
required for fit acceptance.
`scripts/complete_bounded_cpu_covariance.py` performs that continuation after
the minimum process has closed its kernels. It rechecks the minimum and frozen
gradient evidence, saves every Hessian block at the fitted point, reassembles
the complete matrix, and inverts all free curvature entries. Nonpositive,
asymmetric or numerically unstable curvature stays incomplete. The counting
workflow passes the native v11 fit/uncertainty comparison; full HHVV remains
pending.
For local profiles, `run_profile_points.py --engine cpu --bounded-minimum REPORT`
reuses this evaluator across the frozen nuisance grid. It requires an accepted
baseline plan and a completed bounded minimum, then reproduces all 26 reference
arrays exactly before fitting. Only the optimizer's selected nuisance bound is
fixed; every likelihood term and the complete gradient remain evaluated.
The five-point counting profile passes the unchanged native comparison.

GPU measurements use trusted numerical exports translated locally with Combine
11.0.0 / ROOT 6.40.00. The pinned lxplus905 runtime uses
`/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh`
(JAX/jaxlib 0.10.1, CUDA 12 plugin), enables float64 and checks `jax.devices()`
before loading a model. The benchmark process does not import ROOT; its HHVV
Combine reference arrays come from the local v11 stack. A
[separately verified node-local v11/ROOT 6.40 venv](reports/gpu_native_v11_environment_audit.json)
supports native checks through `scripts/run_native_v11_node.sh`, which clears
the inherited ROOT and library paths before activating that environment.

The historical graph benchmark includes local GPU kernels, synchronized host
copies and global NumPy accumulation. Its `scoped` policy stores derivative
executables on node-local disk to bound host memory; standalone call latencies
include loading and cleanup. The untimed gradient validation batch shares each
load across all 26 points. CPU/GPU jobs run sequentially, with hashes, compiler
options, compile/call times and sampled memory recorded in `reports/gpu_*.json`.
Shared GPU activity is recorded separately and limits timing interpretation.
`scripts/compare_local_cpu_hessian.py` verifies every nominal matrix entry
against the separate local CPU reference, reassembling all saved blocks first.
It records the different JAX versions and does not compare times across hosts.
All 26 saved NLL points also pass the
[same-node CPU/GPU and Combine comparison](reports/gpu_hhvv_scoped_value_comparison.json).
The [completed gradient comparison](reports/gpu_hhvv_scoped_gradient_audit.json)
passes all 26 points and the frozen Combine references. Its observed warm medians
are 172.852 s on CPU and 289.252 s on GPU, including scoped loading and cleanup.
The superseded graph measurements are checkpointed; current Asimov work uses
the tensor backend. The running measurement state and all earlier failures are in
[STATUS.md](STATUS.md); small smoke results do not establish full HHVV performance.

HHVV validation scripts explicitly set MX=2000 constant in memory through
`scripts/hhvv_setup.py` and record that configuration in each report. The source
`workspace_MX2000.root` saves MX=125; the generic loader preserves that saved state.
See [STATUS.md](STATUS.md) for the remaining full-likelihood gates.
