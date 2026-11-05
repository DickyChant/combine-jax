"""Framework-neutral ROOT/CERNLIB G110 Landau evaluator and CDF.

The rational coefficient tables and interval definitions are those used by
ROOT MathCore (DENLAN and DISLAN, Kolbig/Schorr, CPC 31 (1984) 97-111).
The CDF is ROOT's separate rational algorithm: its derivative is not replaced
by the PDF. No interpolation table or alternative Landau approximation is used.
Source definitions are in ROOT MathCore
PdfFuncMathCore.cxx / ProbFuncMathCore.cxx and Math/CladDerivator.h.
"""


def _polynomial(coefficients, x):
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = coefficient + x * result
    return result


def _ratio(numerators, denominators, index, x):
    return _polynomial(numerators[index], x) / _polynomial(denominators[index], x)


def standard(v, xp, *, cumulative=False):
    """Literal native branches, with finite inactive arithmetic for AD."""
    numerators, denominators = (CDF_P, CDF_Q) if cumulative else (PDF_P, PDF_Q)
    boundaries = (-5.5, -1., 1., 4. if cumulative else 5., 12., 50., 300.)
    values = []
    # For u < 1e-10 both native exponential expressions underflow exactly to
    # zero. Avoid reciprocal-of-zero residuals on that inactive derivative path.
    left = xp.where(v < boundaries[0], v, -6.)
    u = xp.exp(left + 1.)
    safe_u = xp.where(u >= 1e-10, u, 1e-10)
    correction = ((1., -.4583333333, .6675347222, -1.641741416) if cumulative
                  else (1., .04166666667, -.01996527778, .02709538966))
    scale = xp.sqrt(safe_u) if cumulative else 1. / xp.sqrt(safe_u)
    values.append(xp.where(u >= 1e-10,
        .3989422803 * xp.exp(-1. / safe_u) * scale * _polynomial(correction, safe_u), 0.))
    for index in range(6):
        lo, hi = boundaries[index:index+2]
        point = xp.where((v >= lo) & (v < hi), v, .5*(lo+hi))
        argument = point if index < 3 else 1. / point
        result = _ratio(numerators, denominators, index, argument)
        if index == 0:
            u = xp.exp(-point - 1.)
            scale = 1. / xp.sqrt(u) if cumulative else xp.sqrt(u)
            result = xp.exp(-u) * scale * result
        elif index >= 3 and not cumulative:
            result = argument * argument * result
        values.append(result)
    right = xp.where(v >= 300., v, 400.)
    u = 1. / (right - right * xp.log(right) / (right + 1.))
    values.append(1. - u*(1. + u*(-.4227843351 - 2.043403138*u)) if cumulative
                  else u*u*(1. + u*(-1.845568670 - 4.284640743*u)))
    result = values[-1]
    for boundary, branch in reversed(tuple(zip(boundaries, values[:-1], strict=True))):
        result = xp.where(v < boundary, branch, result)
    return result


def value(x, mean, sigma, xp):
    safe_sigma = xp.where(sigma > 0., sigma, 1.)
    return xp.where(sigma > 0., standard((x-mean)/safe_sigma, xp), 0.)


def integral(low, high, mean, sigma, xp):
    return sigma*(standard((high-mean)/sigma, xp, cumulative=True)
                  - standard((low-mean)/sigma, xp, cumulative=True))

PDF_P = (
    (0.4259894875, -0.124976255, 0.039842437, -0.006298287635, 0.001511162253),
    (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211),
    (0.1788544503, 0.09359161662, 0.006325387654, 6.611667319e-05, -2.031049101e-06),
    (0.9874054407, 118.6723273, 849.279436, -743.7792444, 427.0262186),
    (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.9491),
    (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109),
)

PDF_Q = (
    (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063),
    (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714),
    (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675),
    (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511),
    (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357),
    (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939),
)

CDF_P = (
    (0.2514091491, -0.06250580444, 0.0145838123, -0.002108817737, 0.000741124729),
    (0.2868328584, 0.3564363231, 0.1523518695, 0.02251304883),
    (0.2868329066, 0.3003828436, 0.09950951941, 0.008733827185),
    (1.00035163, 4.503592498, 10.8588388, 7.536052269),
    (1.000006517, 49.09414111, 85.05544753, 153.2153455),
    (1.000000983, 132.9868456, 916.2149244, -960.5054274),
)

CDF_Q = (
    (1.0, -0.005571175625, 0.06225310236, -0.003137378427, 0.001931496439),
    (1.0, 0.6191136137, 0.1720721448, 0.02278594771),
    (1.0, 0.4237190502, 0.1095631512, 0.008693851567),
    (1.0, 5.539969678, 19.33581111, 27.21321508),
    (1.0, 50.09928881, 139.9819104, 420.0002909),
    (1.0, 133.9887843, 1055.990413, 553.2224619),
)
