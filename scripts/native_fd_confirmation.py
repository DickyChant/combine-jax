"""AD-independent confirmation of a native histogram-score FD plateau."""
from native_fd_refinement import REFERENCE_SPREAD

CONFIRMATION_PROBES = 6


def select_confirmed_reference(probes):
    """Require two adjacent three-step windows with a common full spread.

    A short plateau can precede a second change as steps become more local.
    Apply this stricter confirmation to every histogram-score reference; no AD
    value is accepted by this interface or used to choose a window.
    """
    for start in range(len(probes)-CONFIRMATION_PROBES+1):
        window = probes[start:start+CONFIRMATION_PROBES]
        if not all(p['status'] == 'valid' for p in window):
            continue
        if len({(p['stencil'], p['direction']) for p in window}) != 1:
            continue
        if not all(a['step'] > b['step'] for a, b in zip(window, window[1:])):
            continue
        derivatives = [p['derivative'] for p in window]
        spread = (max(derivatives)-min(derivatives))/max(min(map(abs, derivatives)), 1e-8)
        if spread < REFERENCE_SPREAD:
            return dict(status='converged' if min(map(abs, derivatives)) >= 1e-4 else 'roundoff_sensitive',
                selected_first_probe_index=start, full_relative_spread=spread,
                reference_probe_count=CONFIRMATION_PROBES)
    return dict(status='unresolved', reference_probe_count=CONFIRMATION_PROBES)
