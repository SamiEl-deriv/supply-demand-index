import numpy as np
from ....market.stochastic_indices.stochastic_index_base import StochasticIndex
from abc import ABC, abstractmethod

global dt
dt = 1/(365*24*3600)


class ReturnPathGenerator(ABC):
    """
    Paths Generator Base (abstract) class for sub-classing.

    The generate method is resonsible of the stochastic index
    paths construction and it is the one that should be implemented
    in sub-classes.
    """
    def __init__(self, index: StochasticIndex) -> None:
        self.__index = index

    @property
    def index(self):
        return self.__index
    
    @property
    def nbAssets(self):
        return getattr(self.__index, "nbAssets", 1)

    @abstractmethod
    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        """
        Abstract method for generating paths of returns

        Parameters
        ----------
        T     : float
            The path duration (in terms of years).
        N     : int
            The number of paths to draw
        dt    : float
            The timelaps between two steps (expressed in years).
            Default is one second.
        seed  : int
            The seed of the random generator.

        Returns
        -------
        np.ndarray
            The array of cumulated returns . Each row
            is a different path and each column a time step.
        """
        raise NotImplementedError('Implementation required!')

    def replaceDrawingMethod(self) -> None:
        self.__originDrawRetPath = self.__index._drawReturnPath
        self.__index._drawReturnPath = self._drawReturnPath

    def resetDrawingMethod(self) -> None:
        self.__index._drawReturnPath = self.__originDrawRetPath
