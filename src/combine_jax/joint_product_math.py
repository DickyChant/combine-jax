"""Neutral raw Hmumu shape times Chebychev product for native quadrature."""


def value(x, parameters, shape, shape_count, references, xp):
    if shape == 'modz':
        from .modz_math import value as base_value
        base = base_value(x, parameters[:shape_count], *references[0], xp)
    elif shape == 'twoexp':
        from .twoexp_math import value as base_value
        base = base_value(x, *parameters[:shape_count], *references[0], xp)
    else:
        raise ValueError('Unknown joint numerical shape')
    low, high = references[1]
    t = (x-.5*(low+high))/(.5*(high-low))
    polynomial, previous, current = xp.ones_like(x), xp.ones_like(x), t
    for coefficient in parameters[shape_count:]:
        polynomial = polynomial+coefficient*current
        previous, current = current, 2*t*current-previous
    return base*polynomial
