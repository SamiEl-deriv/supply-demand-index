from .yieldcurve import YieldCurve

from typing import Type


class YieldCurveFactory:
    """
    YieldCurveFactory class
    """
    _builders = {}

    @classmethod
    def RegisterYieldCurve(cls, YieldCurveID: str, builder: Type[YieldCurve]):
        """
        Registers a yield curve builder with a given ID

        Parameters
        ----------
        OptionID : str
            The option ID to assign the builder
        """
        cls._builders[YieldCurveID] = builder

    @classmethod
    def CreateYieldCurve(cls, YieldCurveID: str, **kwargs):
        """
        Creates a new yield curve object using a stored builder

        Parameters
        ----------
            OptionID : str
                The yield curve ID for the builder
            **kwargs
                arguments of the yield curve constructor -> float

        Returns
        -------
        yield curve
            yield curve instance

        """

        builder = cls._builders.get(YieldCurveID)
        if not builder:
            raise ValueError(f"No yield curve builder assigned {YieldCurveID}")
        return builder(**kwargs)
