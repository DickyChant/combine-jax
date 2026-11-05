"""Literal RooSumTwoExpPdf values and its finite-range integral (NumPy/JAX)."""


def _zero_integral(slope, low, high, offset, xp):
    # Used only at slope == 0. These terms provide the exact first five
    # limiting derivatives there; no nonzero-slope value uses this polynomial.
    width = high-low
    z = slope*width
    series = 1+z*(.5+z*(1/6+z*(1/24+z*(1/120+z/720))))
    return width*xp.exp(slope*(low-offset))*series


def component(x, slope, low, high, offset, xp):
    denominator = xp.exp(slope*(high-offset))-xp.exp(slope*(low-offset))
    regular = slope/xp.where(slope != 0, denominator, 1.)
    normalizer = xp.where(slope != 0, regular, 1./_zero_integral(slope, low, high, offset, xp))
    return normalizer*xp.exp(slope*(x-offset))


def value(x, a1, a2, fraction, mean, low, high, xp):
    return fraction*component(x, a1, low, high, mean, xp)+(1.-fraction)*component(x, a2, low, high, 0., xp)


def integral(low, high, ref_low, ref_high, a1, a2, fraction, mean, xp):
    def part(slope, offset):
        numerator = xp.exp(slope*(high-offset))-xp.exp(slope*(low-offset))
        denominator = xp.exp(slope*(ref_high-offset))-xp.exp(slope*(ref_low-offset))
        regular = numerator/xp.where(slope != 0, denominator, 1.)
        zero = _zero_integral(slope, low, high, offset, xp)/_zero_integral(slope, ref_low, ref_high, offset, xp)
        return xp.where(slope != 0, regular, zero)
    return fraction*part(a1, mean)+(1.-fraction)*part(a2, 0.)
