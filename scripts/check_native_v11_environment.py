"""Verify the separate pinned-node native reference runtime, without fitting."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


def main():
    directory = Path("/tmp/sqian/combine-jax")
    assert Path.cwd() == directory and platform.node() == "lxplus905.cern.ch"
    prefix = directory/".venv-combine-v11"
    assert Path(sys.prefix) == prefix
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    output = Path("reports/gpu_native_v11_environment.json")
    if output.exists():
        raise FileExistsError("Preserve previous native environment checks")
    started = time.perf_counter()
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  host=platform.node(), workdir=str(Path.cwd()),
                  python=sys.version, executable=sys.executable,
                  prefix=sys.prefix, base_prefix=sys.base_prefix,
                  source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
                      for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(Path("scripts/run_native_v11_node.sh").read_bytes()).hexdigest(),
                  environment_before_import={k: os.environ.get(k) for k in (
                      "PYTHONPATH", "PYTHONNOUSERSITE", "LD_LIBRARY_PATH", "ROOTSYS",
                      "ROOT_INCLUDE_PATH", "JAX_PLATFORMS", "JAX_ENABLE_X64", "TMPDIR")},
                  scope="Separate Combine v11 reference environment only; CPU JAX is an import dependency. No fit, likelihood comparison, or GPU performance acceptance.")
    try:
        assert all(os.environ.get(k) is None for k in ("LD_LIBRARY_PATH", "ROOTSYS", "ROOT_INCLUDE_PATH"))
        assert not any("/lcg/views/" in p or "/site-packages" in p and not Path(p).is_relative_to(prefix)
                       for p in sys.path)
        manifest_path = Path("reports/gpu_native_v11_wheels_compatible.json")
        manifest = json.loads(manifest_path.read_text())
        report["wheel_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report["packages"] = {name: version(name) for name in manifest["wanted"]}
        assert report["packages"] == manifest["wanted"]
        import jax
        devices = jax.devices()
        report.update(jax_version=jax.__version__, devices=[str(d) for d in devices],
                      float64=jax.config.jax_enable_x64)
        assert jax.config.jax_enable_x64 and all(d.platform == "cpu" for d in devices)
        from combine_jax.root_io import load_caching_nll, REFERENCE_FLAGS
        root = load_caching_nll()
        report.update(root_version=str(root.gROOT.GetVersion()), runtime_flags=REFERENCE_FLAGS,
                      combine_version=version("higgs-combine-tool"))
        assert report["root_version"] == "6.40.00" and report["combine_version"] == "11.0.0"
        assert hasattr(root.cacheutils, "CachingSimNLL")
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/" in line.split()[-1] and ".so" in line.split()[-1]})
        native = [p for p in libraries if any(name in p for name in (
            "/ROOT/", "/root.libs/", "/cppyy", "libHiggsAnalysisCombinedLimit"))]
        assert native and all(Path(p).is_relative_to(prefix) for p in native)
        assert any("libCore.so" in p for p in native)
        assert any("libHiggsAnalysisCombinedLimit" in p for p in native)
        assert not any("/lcg/releases/ROOT/" in p or "/lcg/views/" in p for p in libraries)
        report.update(native_libraries=native, loaded_libraries=libraries)
        model_manifest = json.loads(Path("build/gpu_exports/manifest_random.json").read_text())
        report["workspace_copies"] = {}
        for model, path in (("counting", Path("build/gpu_exports/counting.root")),
                            ("hhvv", Path("inputs/workspace_MX2000.root"))):
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            assert sha == model_manifest["models"][model]["source"]["sha256"]
            report["workspace_copies"][model] = dict(path=str(path), sha256=sha, bytes=path.stat().st_size)
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        with output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","root_version","combine_version","jax_version",
                                          "devices","elapsed_seconds","peak_rss_mib")}), flush=True)


if __name__ == "__main__":
    main()
