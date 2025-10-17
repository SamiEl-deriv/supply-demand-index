from typing import Callable
from math import ceil
import numpy as np
import scipy.stats as ss
from .RPG_base import ReturnPathGenerator, dt


class StandardRPG(ReturnPathGenerator):

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        pass

    def replaceDrawingMethod(self) -> None:
        pass

    def resetDrawingMethod(self) -> None:
        pass


class AntitheticRPG(ReturnPathGenerator):

    @property
    def margiRetDist(self) -> Callable:
        return self.index._margiReturnDistrib

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        N_t = int(T/dt)
        half = ceil(N/2)
        arrayShape = (N, N_t, self.nbAssets)
        unifRandNumbArr = np.empty(arrayShape).squeeze()
        unifRandNumbArr[:half] = ss.uniform.rvs(size=(half, *arrayShape[1:]),
                                                random_state=seed)
        unifRandNumbArr[half:] = 1 - unifRandNumbArr[:int(N/2)]
        margRetArr = self.margiRetDist(dt).ppf(unifRandNumbArr)
        cumulRetArr = np.zeros((N, N_t+1, self.nbAssets)).squeeze()
        cumulRetArr[:, 1:] = margRetArr.cumsum(axis=1)
        return cumulRetArr


class StratifiedRPG(ReturnPathGenerator):

    @property
    def cumulRetDist(self) -> Callable:
        return self.index._cumulReturnDistrib

    @property
    def condiRetDist(self) -> Callable:
        return self.index._condiReturnDistrib

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        finalReturnsArray = self._drawFinalReturns(T, N, seed=seed)
        returnsPathsArray = self._bridgeReturnPath(finalReturnsArray, T,
                                                   dt=dt, seed=seed)
        return returnsPathsArray

    def _drawFinalReturns(self, T: float, N: int,
                          seed: int = None) -> np.ndarray:
        uniformRandomArr = ss.qmc.LatinHypercube(self.nbAssets,
                                                 seed=seed).random(N)
        return self.cumulRetDist(T).ppf(uniformRandomArr)

    def _bridgeReturnPath(self, finalReturnsArray: np.ndarray, T: float,
                          dt: float = dt, seed: int = None) -> np.ndarray:
        N, Na, Nt = (*finalReturnsArray.shape, int(T/dt))
        returnsPathsArray = np.empty((N, Nt+1, Na)).squeeze()
        returnsPathsArray[:, 0], returnsPathsArray[:, -1] = 0, finalReturnsArray
        for i in range(Nt, 1, -1):
            returnsPathsArray[:, i-1] = self.condiRetDist((i-1)*dt,
                                                          0, 0,
                                                          returnsPathsArray[:, i],
                                                          i*dt).rvs(random_state=seed)
        return returnsPathsArray


class SobolRPG(ReturnPathGenerator):

    @property
    def cumulRetDist(self) -> Callable:
        return self.index._cumulReturnDistrib

    @property
    def condiRetDist(self) -> Callable:
        return self.index._condiReturnDistrib

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        nbHalfCuts = round(np.log2(T/dt))
        m, Nt, Na = round(np.log2(N)), 2**nbHalfCuts + 1, self.nbAssets
        uniformQMCArr = ss.qmc.Sobol(Nt*Na, seed=seed).random_base2(m)\
            .reshape(-1, Nt, Na).squeeze()
        returnPathArr = np.empty_like(uniformQMCArr)
        returnPathArr[:, 0], returnPathArr[:, -1] = 0, \
            self.cumulRetDist(T).ppf(uniformQMCArr[:, 0])
        # recursive mid-path bridging
        nbPieces, pieceLen, dt_, iQMC = 1, Nt-1, T/Nt, 1
        for _ in np.arange(nbHalfCuts):
            i_mid, i_min, i_max = pieceLen//2, 0, pieceLen
            i = i_mid
            for __ in np.arange(nbPieces):
                condiRetDist = self.condiRetDist(i*dt_,
                                                 returnPathArr[:, i_min],
                                                 i_min*dt_,
                                                 returnPathArr[:, i_max],
                                                 i_max*dt_)
                returnPathArr[:, i] = condiRetDist.ppf(uniformQMCArr[:, iQMC])
                i, i_max, i_min, iQMC = \
                    i+pieceLen, i_max+pieceLen, i_min + pieceLen, iQMC+1
            nbPieces *= 2
            pieceLen = i_mid
        return returnPathArr