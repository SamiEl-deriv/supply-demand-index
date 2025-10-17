
# Import type hinting class
from typing import Type

from ..fdm import FiniteDifference
from .explicit import FDMEulerExplicit
from .implicit import FDMEulerImplicit
from .cranknicolson import FDMEulerCrankNicolson
from .crancknicolsonRannacher import FDMEulerCrankNicolsonRannacher


FDMschemesRefDict : dict[str, Type[FiniteDifference]] = {
    "explicit"       : FDMEulerExplicit,
    "implicit"       : FDMEulerImplicit,
    "crank_nicolson" : FDMEulerCrankNicolson,
    "crank_nicolson_rannacher": FDMEulerCrankNicolsonRannacher
}


class FDMschemeFactory:
    """
    FDMschemeFactory class
    """
    _builders = FDMschemesRefDict

    @classmethod
    def CreateFDM(cls, SchemeId: str):
        """
        Creates a new FDM scheme using a stored builder

        Parameters
        ----------
            SchemeID : str
                The option ID for the builder

        Returns
        -------
        FiniteDifference
            FiniteDifference class instance

        """
        try:
            builder = cls._builders[SchemeId]
        except KeyError:
            raise ValueError("Selected scheme not supported\n"
                             f"Try one of {list(cls._builders.keys())}")
        return builder()
