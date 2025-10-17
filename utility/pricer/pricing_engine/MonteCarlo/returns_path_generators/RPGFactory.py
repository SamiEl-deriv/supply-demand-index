from .RPG_base import ReturnPathGenerator
from .RPG import *

from typing import Type


RPGRefDict : dict[str, Type[ReturnPathGenerator]] = {
    "standard"   : StandardRPG,
    "antithetic" : AntitheticRPG,
    "stratified" : StratifiedRPG,
    "sobol"      : SobolRPG
}

class RPGFactory:
    """
    Paths generator factory class
    """
    _builders = RPGRefDict

    @classmethod
    def CreateRPG(cls, RPGId: str, **kwargs) -> ReturnPathGenerator:
        """
        Creates a new returns paths generator using a stored builder

        Parameters
        ----------
            RPGID : str
                The returns paths generator ID for the builder
            **kwargs
                arguments of the returns paths generator constructor
                NOTE: for now no kwargs needed

        Returns
        -------
        PathsGenrator
            A ReturnPathGenerator instance

        """

        builder = cls._builders.get(RPGId)
        if not builder:
            raise ValueError("No ReturnPathGenerator builder "
                             f"assigned {RPGId}")
        return builder(**kwargs)
