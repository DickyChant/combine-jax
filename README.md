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
RooMultiPdf is rejected unless `--discrete-policy fixed` is explicit; this means
limits conditional on nominal choices, not native discrete-envelope agreement.

The default output is `higgsCombineTest.AsymptoticLimits.mH125.0.root`, containing
a standard TTree named `limit` with five expected quantiles and no observed row.
`--seed` and `--name` follow Combine's filename convention. A matching JSON sidecar
records input/source hashes, dataset identity, fit convergence, evaluation counts,
setup/compile/execution timings, versions and sampled memory. An unresolved
crossing is NaN in the tree and has an explicit sidecar status; it is never a
silent midpoint. Any failed fit fails the command. Stage logs are retained in the
printed work directory. Existing output files are not overwritten.

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

Law bundles and checksums the runtime, requests one H100/A100 and 12 GiB host
memory, fetches the bundle into worker scratch, monitors the job and writes the
ROOT output plus JSON receipt. The unchanged MergeUpperLimits loader produces
its usual NPZ. JAX filenames contain `__jax`, preserving native results.
The existing inference setup supplies and transports the native CMSSW release;
`COMBINE_JAX_NATIVE_CMSSW` is set to that worker's `CMSSW_BASE`. No manual repo
rsync, importer creation, spooled-output transfer or completed-job removal is
needed. Shared software and credentials follow the existing inference workflow.

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
