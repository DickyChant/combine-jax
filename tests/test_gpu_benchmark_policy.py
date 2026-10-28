"""Reject timing comparisons with different executable storage or Hessian policy."""

import json
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_gpu_results import check_execution_policy, main


@pytest.mark.parametrize("changed", ("spool_hessian_executables", "spool_value_gradient",
    "hessian_chunk_size", "spooled_executable_sha256", "scoped_loads",
    "scoped_executable_sha256", "cpu_hessian_no_fusion"))
def test_full_comparison_rejects_mismatched_hessian_policy(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    common = dict(status="passed", source="same", host="same", benchmark_sha256="same",
                  source_sha256="same", constraint_batch_size=32, release_stages=True,
                  trim_compilation_cache=True, spool_hessian_executables=True,
                  spool_value_gradient=True,hessian_chunk_size=3, spooled_executable_sha256="same")
    for backend in ("cpu", "gpu"):
        report = dict(common)
        if backend == "gpu":
            report[changed] = {"spool_hessian_executables":False, "spool_value_gradient":False,"hessian_chunk_size":4,
                               "spooled_executable_sha256":"changed", "scoped_loads":True,
                               "scoped_executable_sha256":"changed", "cpu_hessian_no_fusion":True}[changed]
        Path(f"reports/gpu_counting_{backend}_spooled.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["compare_gpu_results.py", "--report-tag", "spooled", "--allow-partial"])
    # Fail on the policy mismatch before reading arrays or producing a ratio.
    with pytest.raises(AssertionError, match=changed):
        main()
    assert len(list(Path("reports").iterdir())) == 2


def scoped_pair(no_fusion=True):
    common = dict(scoped_loads=True, scoped_executable_sha256="same",
                  spool_hessian_executables=True, spool_value_gradient=True,
                  cpu_hessian_no_fusion=no_fusion, hessian_compiler_options={},
                  timings={operation: dict(term_timings=[dict(compiler_options={})])
                           for operation in ("value", "value_and_grad", "hessian")})
    cpu, gpu = copy.deepcopy(common), copy.deepcopy(common)
    if no_fusion:
        options = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}
        cpu["hessian_compiler_options"] = options
        cpu["timings"]["hessian"]["term_timings"][0]["compiler_options"] = options.copy()
    return cpu, gpu


@pytest.mark.parametrize("no_fusion", (False, True))
def test_scoped_policy_accepts_explicit_backend_options(no_fusion):
    check_execution_policy(*scoped_pair(no_fusion))


def test_scoped_policy_accepts_skipped_empty_hessian():
    cpu, gpu = scoped_pair()
    for report in (cpu, gpu):
        report["timings"]["hessian"]["term_timings"].append(
            dict(skipped_empty_hessian=True, parameters=0, compiler_options={}))
    check_execution_policy(cpu, gpu)


@pytest.mark.parametrize("backend,operation", (("cpu", None), ("gpu", None),
    ("cpu", "hessian"), ("gpu", "hessian"), ("cpu", "value_and_grad"), ("gpu", "value")))
def test_scoped_policy_rejects_unexpected_compiler_options(backend, operation):
    cpu, gpu = scoped_pair()
    report = cpu if backend == "cpu" else gpu
    target = report if operation is None else report["timings"][operation]["term_timings"][0]
    target["hessian_compiler_options" if operation is None else "compiler_options"] = {"unexpected": True}
    with pytest.raises(AssertionError, match="compiler_options"):
        check_execution_policy(cpu, gpu)


@pytest.mark.parametrize("missing", ("--scoped-loads", "--spool-value-gradient",
    "--spool-hessian-executables", "--release-stages", "--trim-compilation-cache"))
def test_scoped_cli_requires_complete_policy(tmp_path, monkeypatch, missing):
    import gpu_benchmark
    monkeypatch.chdir(tmp_path)
    args = ["gpu_benchmark.py", "--model", "counting", "--backend", "cpu",
            "--report-tag", "scoped", "--scoped-loads", "--spool-value-gradient",
            "--spool-hessian-executables", "--release-stages", "--trim-compilation-cache"]
    args.remove(missing)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        gpu_benchmark.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_scoped_cli_preserves_existing_evidence(tmp_path, monkeypatch):
    import gpu_benchmark
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    output = Path("reports/gpu_counting_cpu_scoped.json")
    output.write_text("preserved")
    monkeypatch.setattr(sys, "argv", ["gpu_benchmark.py", "--model", "counting", "--backend", "cpu",
        "--report-tag", "scoped", "--scoped-loads", "--spool-value-gradient",
        "--spool-hessian-executables", "--release-stages", "--trim-compilation-cache"])
    with pytest.raises(FileExistsError):
        gpu_benchmark.main()
    assert output.read_text() == "preserved"
