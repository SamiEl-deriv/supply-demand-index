# Import type hinting class
from typing import Type, Union

from .yieldcurve import YieldCurve
from .yieldcurve_discrete import YieldCurveDiscrete
from .yieldcurve_flat import YieldCurveFlat
from .yieldcurve_linear import YieldCurveLinear


yield_curve_types : dict[str, YieldCurve] = {
    "flat" : YieldCurveFlat,
    "linear" : YieldCurveLinear,
    "step" : YieldCurveDiscrete
}