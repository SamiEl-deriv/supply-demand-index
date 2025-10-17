from typing import Type
import numpy as np
import scipy.stats as ss
from typing import Union
from abc import ABC, abstractmethod


dt = 1 / (365 * 24 * 3600)


class StochasticIndex(ABC):
    """
    Abstract class for stochastic index object.

    Abstract method to be implemented in concrete class
    ---------------------------------------------------
    returns(self: Self, dt: float = dt,
            size: Union[int, tuple[int]] = 1) -> np.ndarray

    '__init__' method and attributes are different for each subclasses
    """

    def drawValuePath(self, S0: float, T: float,
                      N: int = 1, dt: float = dt,
                      seed: int = None) -> np.ndarray:
        return S0 + self._drawReturnPath(T, N=N, dt=dt, seed=seed)

    @abstractmethod
    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Abstract method for generating the one-period (1p) spot returns of the
        asset price according to its stochastic differential equation (SDE).

        Parameters
        ----------
        dt             : float
            Timelapse between two steps in the stochastic process
        size           : int or tuple of int
            size of the array with lines and columns corresponding repectively
            to the different paths and steps of the stochastic process
        random_seed    : int
            seed of the random generator

        Returns
        -------
        np.ndarray
            An array of the spot returns randomly drawn from the
            corresponding index distribution
        """
        raise NotImplementedError('Implementation required!')

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        N_t = int(T/dt)
        returnsPathsArray = np.zeros((N, N_t))
        marginalReturnsArray = self._drawMarginalReturn(size=(N, N_t-1),
                                                        dt=dt, seed=seed)
        returnsPathsArray[:, 1:] = marginalReturnsArray.cumsum(axis=1)
        return returnsPathsArray

    def _margiReturnDistrib(self, dt: float = dt) -> ss.rv_continuous:
        raise NotImplementedError(f'Distribution of {self.__class__.__name__}'
                                  ' marginal returns not implemented')

    def _cumulReturnDistrib(self, T: float,
                            returnsInit: float = 0.0,
                            dt: float = dt) -> ss.rv_continuous:
        raise NotImplementedError(f'Distribution of {self.__class__.__name__}'
                                  ' returns not implemented')

    def _condiReturnDistrib(self, T: float,
                            returns1: np.ndarray, T1: float,
                            returns2: np.ndarray, T2: float,
                            dt: float = dt) -> ss.rv_continuous:
        raise NotImplementedError(f'Distribution of {self.__class__.__name__}'
                                  ' returns, conditionnally to past and future'
                                  ' returns, not implemented')

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        raise NotImplementedError('Expected forward value of '
                                  f'{self.__class__.__name__} not implemented')

    def __repr__(self) -> str:
        """
        Dunder method for representing index object
        """
        return f'{self.__class__.__name__}{tuple(self.__dict__.values())}'


def geometricIndexDecorator(cls: Type[StochasticIndex]) -> Type[StochasticIndex]:

    def drawValuePath_new(self, S0: float, T: float,
                          N: int = 1, dt: float = dt,
                          seed: int = None) -> np.ndarray:
        returnPath = self._drawReturnPath(T, N, dt, seed)
        valuePath = S0 * np.exp(returnPath)
        return valuePath

    cls.drawValuePath = drawValuePath_new
    return cls
