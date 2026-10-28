"""MultiDimFit TTrees read by unmodified dhi scan and impacts loaders."""
from pathlib import Path


def write_multidim_tree(path, rows, *, mass=125., seed=0, wall_seconds=0.):
    import numpy as np
    import uproot
    if not rows or rows[0]['quantile'] != -1 or rows[0]['deltaNLL'] != 0:
        raise ValueError('First row must be the global best fit')
    names = tuple(rows[0]['parameters'])
    fields = dict(limit='float64', limitErr='float64', mh='float64',
        syst='int32', iToy='int32', iSeed='int32', iChannel='int32',
        t_cpu='float32', t_real='float32', quantileExpected='float32',
        deltaNLL='float32', nll0='float64', nll='float64')
    if set(names) & set(fields):
        raise ValueError('Parameter name collides with standard Combine output branch')
    fields.update({name: 'float32' for name in names})
    count = len(rows)
    arrays = {n: np.zeros(count, dtype=d) for n, d in fields.items()}
    for i, row in enumerate(rows):
        if set(row['parameters']) != set(names):
            raise ValueError('Inconsistent output parameters')
        for name, value in row['parameters'].items():
            arrays[name][i] = value
        for name in ('deltaNLL', 'nll0', 'nll'):
            arrays[name][i] = row[name]
        arrays['quantileExpected'][i] = row['quantile']
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite MultiDimFit output')
    arrays['mh'][:] = mass
    arrays['iSeed'][:] = seed
    arrays['syst'][:] = 1
    arrays['t_real'][:] = wall_seconds
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with uproot.recreate(temporary) as handle:
        handle.mktree('limit', fields).extend(arrays)
    temporary.replace(path)
