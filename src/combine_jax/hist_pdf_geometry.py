"""Framework-neutral exact partial integrals of fixed RooHistPdf tables."""
import numpy as np

from .graph import UnsupportedModelError


def integral_table(name, attributes, axes, bounds):
    """Integrate selected bin widths; preserve signed bins and native full sum.

    A remaining-coordinate lookup clamps to the first/last histogram bin, as
    RooDataHist::sum does. Point evaluation separately clips negative bins and
    returns zero outside the grid; those operations must not be used here.
    """
    selected = set(axes)
    names = attributes['axes']; edges = attributes['edges']
    if not selected <= set(names):
        raise UnsupportedModelError(f'{name}: incompatible RooHistPdf integral over {axes}')
    for axis, edge in zip(names, edges, strict=True):
        if axis in selected:
            low, high = bounds[axis]
            if low > edge[0] or high < edge[-1]:
                raise UnsupportedModelError(f'{name}: restricted RooHistPdf integral range')
    if selected == set(names):
        return (), (), np.asarray(attributes['sum_weights'], dtype=np.float64)
    result = np.asarray(attributes['values'], dtype=np.float64)
    remaining_names, remaining_edges = list(names), list(edges)
    for dimension in reversed(range(len(names))):
        if names[dimension] not in selected: continue
        shape = [1]*result.ndim; shape[dimension] = len(edges[dimension])-1
        result = np.sum(result*np.diff(edges[dimension]).reshape(shape), axis=dimension)
        del remaining_names[dimension]; del remaining_edges[dimension]
    return tuple(remaining_names), tuple(remaining_edges), result
