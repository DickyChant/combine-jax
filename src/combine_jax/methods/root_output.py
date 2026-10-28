"""Standard expected-only Combine output, without loading a second ROOT stack."""
import math
from pathlib import Path

from .asymptotic import QUANTILES, expected_nll_crossing


def audit_crossings(result):
    """Never publish a geometric midpoint without an evaluated crossing bracket."""
    rows = []
    for q, value, reason in zip(result['quantiles'], result['limits'],
                                result['crossing_reasons'], strict=True):
        trace = [row for row in result['trace'] if row['quantile'] == q]
        target = expected_nll_crossing(q, result['confidence_level'])
        deltas = [row['delta_nll'] for row in trace]
        bracketed = bool(deltas) and min(deltas) <= target <= max(deltas)
        accuracy = .05 * result['options']['minimizer_tolerance']
        nll_match = any(row['mu'] == value and abs(row['delta_nll']-target) < accuracy
                        for row in trace)
        resolved = (math.isfinite(value) and value >= 0 and
                    ((reason == 'nll_accuracy' and nll_match) or bracketed))
        rows.append(dict(quantile=q, limit=float(value) if resolved else None,
            status='resolved' if resolved else 'unresolved_crossing',
            crossing_reason=reason, evaluated_bracket=bracketed,
            nll_accuracy_passed=nll_match, target_delta_nll=target,
            diagnostic_candidate=float(value)))
    return rows


def write_limit_tree(path, rows, *, mass=125., seed=0, wall_seconds=0.):
    """Write a TTree (not an RNTuple) compatible with dhi's unchanged loader.

    The five expected rows are ordered as native Combine. Unresolved rows are
    NaN, with reasons in the required caller's JSON sidecar. No observed row.
    """
    import numpy as np
    import uproot
    if tuple(row['quantile'] for row in rows) != QUANTILES:
        raise ValueError('Exactly the five expected quantiles are required')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    fields = dict(limit='float64', limitErr='float64', mh='float64',
        syst='int32', iToy='int32', iSeed='int32', iChannel='int32',
        t_cpu='float32', t_real='float32', quantileExpected='float32')
    arrays = {name: np.zeros(5, dtype=dtype) for name, dtype in fields.items()}
    arrays.update(limit=np.array([r['limit'] if r['limit'] is not None else np.nan for r in rows]),
        limitErr=np.full(5, np.nan), mh=np.full(5, float(mass)),
        syst=np.ones(5, dtype=np.int32), iSeed=np.full(5, seed, dtype=np.int32),
        t_real=np.full(5, wall_seconds, dtype=np.float32),
        quantileExpected=np.array(QUANTILES, dtype=np.float32))
    temporary = path.with_name(path.name+'.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(temporary) as handle:
        handle.mktree('limit', fields).extend(arrays)
    temporary.replace(path)
