"""Exhaustive native FD refinement, with no AD-dependent step selection."""
from native_fd_scan import FRACTIONS, scan_coordinate


REFINEMENT_FRACTIONS=(1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,*FRACTIONS)
REFERENCE_SPREAD=1e-5


def select_reference(probes):
    """First three distinct, same-stencil steps with tighter full spread."""
    for start in range(len(probes)-2):
        triple=probes[start:start+3]
        if not all(p['status']=='valid' for p in triple):continue
        if len({(p['stencil'],p['direction']) for p in triple})!=1:continue
        if not all(a['step']>b['step'] for a,b in zip(triple,triple[1:])):continue
        derivatives=[p['derivative'] for p in triple]
        spread=max(derivatives)-min(derivatives)
        relative=spread/max(min(map(abs,derivatives)),1e-8)
        if relative<REFERENCE_SPREAD:
            return dict(status='converged' if min(map(abs,derivatives))>=1e-4 else 'roundoff_sensitive',
                        selected_first_probe_index=start,full_relative_spread=relative)
    return dict(status='unresolved')


def refine_coordinate(variable,value,*,scale,lower,upper,zero):
    if lower==upper:return dict(coordinate=variable.getVal(),probes=[],status='fixed_bounds')
    probes=[];coordinate=float(variable.getVal())
    for fraction in REFINEMENT_FRACTIONS:
        case=scan_coordinate(variable,value,scale=scale,lower=lower,upper=upper,zero=zero,fractions=(fraction,))
        assert len(case['probes'])==1 and variable.getVal()==coordinate
        probes.extend(case['probes'])
    return dict(coordinate=coordinate,probes=probes,**select_reference(probes))
