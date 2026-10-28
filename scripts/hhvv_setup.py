"""Explicit benchmark configuration; the generic backend preserves caller state."""


def configure_hhvv(workspace, mass=2000.):
    mx = workspace.var("MX")
    if not mx:
        raise ValueError("HHVV workspace has no MX variable")
    if not mx.getMin() <= mass <= mx.getMax():
        raise ValueError(f"Mass {mass} is outside the saved MX range")
    mx.setVal(float(mass))
    mx.setConstant(True)
    return {"MX": {"value": float(mx.getVal()), "constant": bool(mx.isConstant())}}
