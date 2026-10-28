"""Independent native five-point derivatives with fixed, AD-free selection."""
import math


FRACTIONS = (1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5,5e-6,2e-6,1e-6,5e-7,2e-7,1e-7)


def scan_coordinate(variable, value, *, scale, lower, upper, zero, fractions=FRACTIONS):
    coordinate = float(variable.getVal())
    case = dict(coordinate=coordinate, probes=[])
    if lower == upper:
        return dict(case,status='fixed_bounds')
    left,right = coordinate-lower,upper-coordinate
    for fraction in fractions:
        step = fraction*scale
        central = min(left,right)>=2*step
        direction = 1. if central or right>=left else -1.
        if not central: step = min(step,max(left,right)/5)
        probe = dict(step=step,fraction=fraction,stencil='central' if central else 'forward',
                     direction=direction,zero_point_nll_values=[])
        try:
            assert step>0 and coordinate+direction*step!=coordinate
            for multiple in ((-2,-1,1,2) if central else (1,2,3,4)):
                shifted=coordinate+direction*multiple*step
                assert lower<=shifted<=upper
                variable.setVal(float(shifted))
                probe['zero_point_nll_values'].append(value())
            values=probe['zero_point_nll_values']
            derivative=((8*(values[2]-values[1])-(values[3]-values[0]))/(12*step) if central else
                sum(c*(v-zero) for c,v in zip((48,-36,16,-3),values,strict=True))/(12*direction*step))
            assert math.isfinite(derivative)
            probe.update(status='valid',derivative=derivative)
        except (ValueError,AssertionError) as exc:
            probe.update(status='invalid',error=repr(exc))
        finally:
            variable.setVal(coordinate); value()
        case['probes'].append(probe)
        triple=case['probes'][-3:]
        if len(triple)==3 and all(p['status']=='valid' for p in triple):
            changes=[abs(a['derivative']-b['derivative'])/max(abs(b['derivative']),1e-8)
                for a,b in zip(triple,triple[1:])]
            if max(changes)<1e-4:
                case.update(status='converged' if min(abs(p['derivative']) for p in triple)>=1e-4 else 'roundoff_sensitive',
                    selected_first_probe_index=len(case['probes'])-3,adjacent_relative_changes=changes)
                break
    case.setdefault('status','unresolved')
    return case
