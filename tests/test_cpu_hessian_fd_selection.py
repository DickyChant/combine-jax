"""Reference-only convergence selection for the supplemental compiler probe."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from refine_cpu_hessian_fd import first_converged_triple


def test_requires_the_endpoints_to_converge_too():
    # Adjacent pairs alone pass, but the first triple's endpoints do not.
    values = [0., .000075, .00015, .000151, .000152]
    result = first_converged_triple(np.asarray(values)[:, None, None], np.ones(1))
    assert result["indices"] == (1, 2, 3)


def test_keeps_first_converged_triple_even_when_later_triples_are_tighter():
    values = [1., 1.00001, 1.00002, 9., 9., 9.]
    result = first_converged_triple(np.asarray(values)[:, None, None], np.ones(1))
    assert result["indices"] == (0, 1, 2)


def test_no_converged_triple_is_explicit():
    values = [1., 2., 3., 4., 5.]
    assert first_converged_triple(np.asarray(values)[:, None, None], np.ones(1)) is None
