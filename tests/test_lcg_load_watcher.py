from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from watch_lcg_recovery_load import matches, split_compute


def current():
    return dict(pid=10, state="S", start_ticks="1234",
                cmdline="python scripts/measure_lcg_recovered_hessian.py --output reports/result.json ")


def test_exact_process_identity():
    assert matches(current(), ticks=1234, script="scripts/measure_lcg_recovered_hessian.py")


@pytest.mark.parametrize("change", ["missing", "zombie", "reused", "substring", "different_script"])
def test_reject_unverified_process_identity(change):
    process = current()
    if change == "missing":
        process = None
    elif change == "zombie":
        process["state"] = "Z"
    elif change == "reused":
        process["start_ticks"] = "1235"
    elif change == "substring":
        process["cmdline"] = "python unrelated_scripts/measure_lcg_recovered_hessian.py"
    elif change == "different_script":
        process["cmdline"] = "python scripts/gpu_benchmark.py"
    assert not matches(process, ticks=1234, script="scripts/measure_lcg_recovered_hessian.py")


def test_only_verified_live_benchmarks_count_as_ours():
    processes = [dict(pid=10, memory_mib=120.), dict(pid=20, memory_mib=1400.)]
    ours, others = split_compute(processes, {10: current()})
    assert ours == processes[:1] and others == processes[1:]
    assert split_compute(processes, {}) == ([], processes)
