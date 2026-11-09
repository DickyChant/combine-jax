# combine-jax

Statistical inference on a float64 JAX likelihood imported from CMS Combine
workspaces. Asymptotic expected limits are the first method; the existing graph
backend and native comparisons remain the numerical oracle. Unsupported classes
raise errors. MultiDimFit, FitDiagnostics, Impacts and toys are not yet a complete
Combine replacement. See [DESIGN.md](DESIGN.md) for conventions and limitations.

## Install an isolated runtime

On an EL9 CERN node with CVMFS, start from this public checkout and an installed
**Combine v10.3.3 / CMSSW_14_1_0_pre4 / ROOT 6.30** release. No private wheel,
model artifact or validation report is required. The native release is read-only.
Choose a node-local or other sufficiently large runtime prefix (about 2 GB).
The source build requires a C++17 compiler, Boost and Eigen headers, available on
lxplus. Internet access is needed at installation time only.

```bash
git clone https://github.com/DickyChant/combine-jax.git
cd combine-jax
bash scripts/setup_runtime.sh \
  --prefix /your/scratch/combine-jax-runtime \
  --native-cmssw /your/software/CMSSW_14_1_0_pre4
source /your/scratch/combine-jax-runtime/setup.sh
```

The installer builds `higgs-combine-tool` from a pinned public source revision
in an isolated Python 3.13 importer venv, with pip ROOT 6.40. It packages the code,
venv, checksums and launcher into `runtime.tar.gz`. It uses the exact CUDA view
`LCG_110a_cuda/x86_64-el9-gcc13-opt` for execution (JAX 0.10.1). These ROOT stacks
never share an interpreter. To update this checkout's code while retaining the
installed dependencies, repeat with `--refresh-code`. Rebundle before submitting.
The prefix can be relocated on workers; CVMFS must be mounted there too.

## Import, fit and scan a HIG-23-006 workspace

The public package now supplies `import`, `fit` and `scan`; they need no
`private_tools/` files. Use the runtime above (or refresh its code and rebundle).
`import` runs in its isolated pip ROOT environment; `fit` and `scan` run ROOT-free
in the CUDA view. A model is a directory containing checksummed JSON and numeric
NPZ arrays, not a pickled workspace. Full HIG import/compilation needs a large
CPU worker; do not load it on a small laptop. Inputs are read-only.

On the CPU import worker, set `HIG_WORKSPACE` to your accessible compiled kl
snapshot and run:

```bash
combine-jax import "$HIG_WORKSPACE" --preset hig23006-kl \
  --analysis hig23006 --data toys/toy_asimov --discrete fixed -o model
combine-jax fit model --device cpu --threads 2 -o fit.json
combine-jax scan model --device cpu --threads 2 --poi kl --points -15 1 10 26 -o scan.json
```

For CPU fits, reserve the worker CPUs and RAM before starting. `--threads N`
selects the first N **logical CPUs in the existing affinity**, without widening
a scheduler or `taskset` allocation. It sets OMP/OpenBLAS/MKL and XLA CPU pool
configuration before NumPy/JAX starts. It overrides `COMBINE_JAX_CPU_THREADS=N`;
without either setting, the caller's affinity and environment are preserved.
The runtime launcher preserves caller `XLA_FLAGS` and thread variables across
LCG setup; its fallback pool settings remain one thread. Unrelated XLA flags
are retained when `--threads` overrides CPU settings. This applies to CPU
`fit`/`scan` and `limit`/`limits`/`multidim` (`--backend cpu` for those methods).
The JSON receipt records `cpu_threads` (request, source, affinity and exact
environment), alongside measured process thread counts and memory. Requested
threads are not a guarantee that every kernel runs concurrently on N CPUs.

For example, on a worker allocated 16 CPUs and sufficient RAM:

```bash
combine-jax fit model --device cpu --threads 16 \
  --host-memory-limit-gib 30 -o fit_cpu16.json
# Equivalent default for subsequent CPU commands:
export COMBINE_JAX_CPU_THREADS=16
combine-jax scan model --device cpu --poi kl --points -15 1 10 26 \
  --host-memory-limit-gib 30 -o scan_cpu16.json
```

These options do not request scheduler resources or change precision, EDM or
native-validation tolerances. Full-model timing depends on the worker and model;
run the same dataset and settings when comparing CPU and GPU execution.

The preset selects snapshot `MultiDimFit`, mass 125.38, kl range [-15,26], other
couplings at SM, the README's attribute freezes and runtime definitions, and
explicit v10-density/ROOT-6.30 CB conventions. It preserves the snapshot's kl
starting value. Generic workspaces can use `--snapshot`, `--poi`, `--set`,
`--freeze`, `--ranges`, `--mass`, `--freeze-attributes` and repeated `--X-rtd`
instead of a preset. Changing the POI or folded couplings requires a new import.
In `fit`/`scan`, `--set` changes starting values; `--freeze` fixes named parameters.

For HIG's authorized observed study, import into a separate directory with
`--analysis hig23006 --data data_obs`. Each report records that selection and
dataset identity. HHVV/other blind-analysis policies refuse observed inputs.
The native HIG gradient gate remains open; these commands do not certify it or
claim a native-identical minimizer trajectory. Failed fits leave a report and
return nonzero; a failed global minimum prevents conditional scan fits.
The public `fit`/`scan` iteration cap is **10,000** (also selectable with
`--max-iterations 10000`). HIG observed fits can exhaust the former 1,000-step
default. Raising the cap does not relax the raw-NLL EDM threshold of `1e-4`
or guarantee convergence. Errors name the failed fit, stopping reason, iteration
count and EDM; an invalid scan denominator identifies the lower conditional fit.
Structured details are saved under `failures` in the report.
If a converged conditional point lies below the unrestricted fit, `scan`
automatically retries the unrestricted fit once from the lowest converged point.
This applies to fixed and profiled discrete choices, preserving explicit freezes.
Both attempts are retained under `fits`; `global_recovery` records the seed and
improvement. Use `scan_points` for likelihood differences against the selected
denominator. Historical `fits[].deltaNLL` values retain their explicitly named
original denominator. A failed recovery or remaining lower point still returns
nonzero; estimated EDM alone does not certify a global minimum.

Host RAM and device allocations are separate checks: the compiler guard is
8 GiB, while the default host guard is a **lifetime peak RSS of 11,264 MiB**
(11 GiB), including compilation and Python/NumPy memory. `fit` and `scan` accept
`--host-memory-limit-gib` when the allocated worker RAM permits a larger budget:

```bash
combine-jax scan model --device cpu --poi kl --points -15 1 10 26 \
  --host-memory-limit-gib 30 -o scan.json
```

The Python API is `run_model(..., host_memory_limit_gib=30)`. This only changes
the host guard; it does not change float64, convergence, the 8 GiB compiler
guard, or the batch scheduler's memory allocation. Reserve enough worker RAM
separately. The end-to-end `limits` and `multidim` commands accept the same option
and apply it to both their process-group monitor and fit worker. Reports retain
the requested limit and measured peak; invalid/nonpositive limits are refused.
Reports distinguish current RSS,
lifetime peak, anonymous/file-backed pages, thread count and sampling stage.
Fewer CPU threads do not imply a lower host-memory peak; comparisons require
identical model/runtime hashes and stage measurements.

Shared-shape RooMultiPdf studies can opt into native-order batched profiling:

```bash
combine-jax import "$HIG_WORKSPACE" --preset hig23006-kl \
  --analysis hig23006 --data toys/toy_asimov --discrete profile \
  --discrete-profile-mode native-batched -o profiled_model
combine-jax scan profiled_model --device gpu --poi kl --points -15 1 10 26 \
  --candidate-batch-size 4 -o profiled_scan.json
```

This mode requires the workspace's registered category order. Reports retain
category traces, physical NLL and native masking offsets separately. Conditional
fits start from the global fit's values and categories. A lower conditional point
triggers one global recovery attempt; `scan_points` names the selected denominator.
The existing independent-shape mode remains the default. Native fixture checks
and full-HIG validation are separate; this option does not claim matching native
intermediate category choices or a certified full-HIG discrete result.

For an allocated H100/A100, replace `--device cpu` by `--device gpu`. The example
[`scripts/example_gpu_scan.sub`](scripts/example_gpu_scan.sub) transfers a runtime
and model archive and returns timing, convergence, memory and fit captures:

```bash
# On the preparation/submission host, not a memory-limited laptop:
cp "$DHI_COMBINE_JAX/runtime.tar.gz" runtime.tar.gz
tar -czf model.tar.gz model
condor_submit scripts/example_gpu_scan.sub
```

Edit the example's `arguments` to pass the exact scan grid. The published
83-point grid is -15,-14.5,...,26; an observed comparison should use the exact
HEPData point list. Fits use float64, corrected estimated EDM <1e-4 and the
8 GiB static allocation guard. Reports separate load/setup, tracing, compilation
and execution. A persistent `--cache` is optional. The LCG_110a CUDA view currently
selects ptxas 12.5.40 and triggers XLA's CUDA <=12.6.2 clamp warning; newer-compiler
and CPU fit comparisons remain necessary before a hardware agreement claim.

`--discrete profile` imports the existing independent-shape profiling path.
HIG's shared-category shapes still require the native-order batched extension;
that import fails explicitly until supported. The validated fixed-choice study
continues separately. Likewise, an r/r_gghh limit needs a separately prepared
background-only Asimov model with the selected strength floating: a kl-scan
model with those strengths folded is insufficient. The existing public
`combine-jax limit --model BACKGROUND_MODEL --poi r --backend gpu --output limit.json`
and `limits` commands retain their provenance checks; HIG-specific preparation
and limit validation are still in progress.

## Blinded end-to-end limits

```bash
combine-jax limits --workspace workspace.root --poi r_hhvv --mass 125 \
  --setParameters MX=4000 --freezeParameters MX \
  --run expected --noFitAsimov --backend gpu
```

`--backend gpu` requires one allocated full H100 or A100; never run this on a
login node. Use `--backend cpu` for small checks. CPU import and GPU fitting run
in separate processes. `--setParameterRanges 'r_hhvv=0,20:MX=1000,6000'` overrides
workspace ranges; without it they are preserved. `--rMax` changes the initial
POI range used by the native stepping procedure, which can expand its bracket.
Only one POI is supported. `--run blind` also selects pre-fit background Asimov.
`--run expected` without `--noFitAsimov` is rejected: this entrypoint never fits
real data to construct an Asimov dataset. There is no observed mode.

The native v10 process generates the pre-fit background-only Asimov, with nominal
nuisances/global observables and the requested mass. The importer consumes those
same rows using the corrected `v10_density` template convention. JAX uses the
`structural_sequential` gradient and device-resident BFGS/EDM fit, retaining
workspace nuisance bounds and an 8 GiB XLA static allocation guard.
RooMultiPdf requires an explicit policy. `--discrete-policy profile` runs the
native short-combination ordering and stored correction penalties with on-device
continuous fits. Independent, unconstrained alternative-shape parameters are
supported; shared/constrained shapes and nested MultiPdfs raise errors pending
validation. `--discrete-policy fixed` instead reports limits conditional on the
nominal choices and is not a discrete envelope. See DESIGN.md for the exact
stopping conventions and the distinction between BFGS and native Minuit2.

The default output is `higgsCombineTest.AsymptoticLimits.mH125.0.root`, containing
a standard TTree named `limit` with five expected quantiles and no observed row.
`--seed` and `--name` follow Combine's filename convention. A matching JSON sidecar
records input/source hashes, dataset identity, fit convergence, evaluation counts,
setup/compile/execution timings, versions and sampled memory. An unresolved
crossing is NaN in the tree and has an explicit sidecar status; it is never a
silent midpoint. Any failed fit fails the command; its last fit checkpoint is
retained in the sidecar when available. Stage logs are retained in the printed
work directory. Existing output files are not overwritten.

## DHI / law integration

The separate inference `jax_backend` branch provides an opt-in backend; native
Combine remains the default. After the usual inference setup and the runtime's
`setup.sh`, use:

```bash
law run MergeUpperLimits \
  --datacards /path/to/datacard.txt --version expected_validation \
  --hh-model hhvv_model.model_default --pois r_hhvv \
  --scan-parameters MX,4000,4000,1 --limit-backend jax \
  --UpperLimits-workflow htcondor --workers 1
```

Law bundles and checksums the runtime, requests one H100/A100 and eight CPU slots
(CERN assigns 24 GB host memory), fetches the bundle into worker scratch, monitors the job and writes the
ROOT output plus JSON receipt. The unchanged MergeUpperLimits loader produces
its usual NPZ. JAX filenames contain `__jax`, preserving native results.
For discrete profiling, add `--jax-discrete-policy profile`; filenames also
contain `profilediscretes`. JAX worker logs use a separate `jax_<policy>` directory
so paired native/JAX validations retain both logs.
The existing inference setup supplies and transports the native CMSSW release;
`COMBINE_JAX_NATIVE_CMSSW` is set to that worker's `CMSSW_BASE`. No manual repo
rsync, importer creation, spooled-output transfer or completed-job removal is
needed. Shared software and credentials follow the existing inference workflow.

## Experimental Asimov scans and impacts

`combine-jax multidim` (also `-M MultiDimFit`) provides grid scans, profile
intervals and nuisance impacts. Counting-card comparisons against native v10
pass. A four-point 1D Asimov scan also passes through law against native v10;
2D and nuisance-impact comparisons are ongoing. These methods
require `--toys -1` and an explicit generation value for every POI. They do not
read observed data or use a post-fit snapshot.

```bash
combine-jax multidim --workspace workspace.root --poi r --toys -1 \
  --setParameters r=0 --setParameterRanges r=0,3 \
  --algo grid --gridPoints 4 --backend gpu

combine-jax multidim --workspace workspace.root --poi r --toys -1 \
  --setParameters r=0 --setParameterRanges r=0,3 \
  --algo batch-impacts --parameters r,lumi,background_norm --backend gpu \
  --output impacts.root
```

Use the parameter names from the actual workspace and ranges where the native
likelihood is defined. Invalid points remain explicit failures. Two-dimensional grids accept
`--poi first,second`, one range and grid count per axis, and `--firstPoint` /
`--lastPoint` for partitioning. Each conditional fit starts from the saved global
minimum. Batched impacts reuse the import, global fit and conditional executable;
their output is a directory of standard `limit` TTrees plus a JSON parameter-to-file
map. The default EDM validation uses the exact final AD Hessian. Flat global POI
directions require an explicit `--global-edm-validation estimated` choice and
independent validation; that choice does not certify global covariance.

`--saveWorkspace` works as in Combine: the output file also holds workspace `w`
with a `MultiDimFit` snapshot of the combine-jax global fit, and the generated
Asimov dataset. Native Combine can then start from that fit:

```bash
combine-jax -M MultiDimFit -d workspace.root --poi r --toys -1 --setParameters r=1 \
  --algo singles --saveWorkspace --output higgsCombineJax.MultiDimFit.mH125.0.root
combine -M MultiDimFit -d higgsCombineJax.MultiDimFit.mH125.0.root -w w \
  --snapshotName MultiDimFit -D combine_jax_asimov_nominal ...
```

The snapshot holds every workspace variable and category, as Combine's own does:
fitted values for the floating parameters, unchanged nominal values for global
observables and frozen parameters. Use `-D combine_jax_asimov_nominal` rather than
`-t -1`, which would regenerate the Asimov from the snapshot. For a pre-fit
Asimov the global fit is the nominal point, so the snapshot gains little there;
it pays off for fits whose minimum is far from the starting values.
`--saveWorkspace` is not yet available with `--discrete-policy profile` or
`--algo batch-impacts`.

The inference branch also supplies `--scan-backend jax` and
`--impacts-backend jax`, with separate targets and unchanged merge/plot loaders:

```bash
law run MergeLikelihoodScan \
  --datacards /path/to/datacard.txt --version asimov_scan \
  --hh-model hhvv_model.model_default --pois r_hhvv \
  --parameter-values MX=4000:r_hhvv=0 --frozen-parameters MX \
  --scan-parameters r_hhvv,0,3,4 --scan-backend jax \
  --LikelihoodScan-workflow htcondor --workers 1

law run MergePullsAndImpacts \
  --datacards /path/to/datacard.txt --version asimov_impacts \
  --hh-model hhvv_model.model_default --pois r_hhvv \
  --parameter-values MX=4000:r_hhvv=0 --frozen-parameters MX \
  --parameter-ranges r_hhvv=0,10 --impacts-backend jax \
  --PullsAndImpacts-workflow local --JAXImpactBatch-workflow htcondor --workers 1
```

`--jax-batch-size` controls the number of nuisance intervals sharing one GPU job
(default 32). `--only-parameters` selects a smaller validation set. The local
PullsAndImpacts tasks copy the finished batch files; the fits run on the requested
HTCondor GPU. `PlotPullsAndImpacts` consumes the same merged result. Full covariance
matrices, snapshot fits and discrete-component nuisance impacts are not supplied
by this implementation.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,output]'
# ROOT-dependent tests additionally need the pinned public Combine importer.
source scripts/env.sh
pytest -q tests/test_asymptotic_statistics.py tests/test_runtime_limits.py
```

Legacy benchmark scripts are research utilities, not installation prerequisites.
Agreement and performance are dataset/configuration dependent; validate each new
model against its native reference before making a numerical or speed claim.

## License

Apache License 2.0, the same as CMS Combine. See [LICENSE](LICENSE).
