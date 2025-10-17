from typing import Type, Union
import numpy as np
from collections import OrderedDict
from .stochastic_index_base import StochasticIndex, geometricIndexDecorator, dt
from .stochastic_index import VolIndex
from .utils import norm_multivariate, norm_matrix
from scipy.stats import rv_continuous, matrix_normal


@geometricIndexDecorator
class VolBasket(StochasticIndex):

    """
    Multi-Volatility index class.

    This class represents a collection of volatility indices with correlated
    returns.
    """

    def __init__(self,
                 corr: tuple[float] = 0,
                 **assets: StochasticIndex) -> None:
        """
        Parameters
        ----------
        corr : tuple[float], optional
            The correlation coefficients between the volatility indices.
            Default is 0.
        **assets : StochasticIndex
            The volatility indices included in the MultiVolIndex object.
            The keys represent the names of the indices, and the values
            represent the corresponding StochasticIndex objects.
        """
        corr_matrix = np.eye(len(assets))
        i_x, i_y = np.tril_indices_from(corr_matrix, k=-1)
        try:
            corr_matrix[i_x, i_y] = corr_matrix[i_y, i_x] = corr
        except ValueError:
            raise TypeError('Incorrect number of correlation coefficients')

        try:
            vols = np.array([asset.volatility for asset in assets.values()])
            drifts = np.array([asset.drift for asset in assets.values()])
        except AttributeError:
            raise TypeError('MultiVolIndex object should include only VolIndex'
                            ' object')

        self.__assets: OrderedDict[StochasticIndex] = OrderedDict(**assets)
        self.__nbAssets: float = len(assets)
        self.__corr: np.ndarray = corr_matrix
        self.__vols: np.ndarray = vols
        self.__drifts: np.ndarray = drifts

    def _drawMarginalReturn(self, dt: float = dt,
                            size: Union[int, tuple[int]] = 1,
                            seed: int = None) -> np.ndarray:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """
        margiReturnArr = self._margiReturnDistrib(dt=dt).rvs(size=size,
                                                             random_state=seed)
        return margiReturnArr.reshape((size[0], size[1], self.__nbAssets))

    def _margiReturnDistrib(self, dt: float = dt) -> rv_continuous:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """
        return norm_multivariate(mean=self.drift*dt, cov=self.cov*dt)

    def _cumulReturnDistrib(self, T: float,
                            returnsInit: Union[float, np.ndarray] = 0.0
                            ) -> rv_continuous:
        """
        Final cumulated returns distribution

        Parameters
        -------
        T        : float
            Time length of the stochastic process
        Returns
        -------
        ss.rv_continuous
            The final cumulated returns distribution at time t
        """
        return norm_multivariate(mean=self.drift*T + returnsInit,
                                 cov=self.cov*T)

    def _condiReturnDistrib(self, T: float,
                            return1: np.ndarray, T1: float,
                            return2: np.ndarray, T2: float
                            ) -> rv_continuous:
        mean = ((T2-T)*return1 + (T-T1)*return2) / (T2-T1)
        cov = self.cov * (T2-T) * (T-T1) / (T2-T1)
        return norm_matrix(mean=mean, colcov=cov)

    def _drawReturnPath(self, T: float,
                        N: int = 1, dt: float = dt,
                        seed: int = None) -> np.ndarray:
        N_t = int(T/dt)
        Na = self.__nbAssets
        returnsPathsArray = np.zeros((N, N_t, Na))
        marginalReturnsArray = self._drawMarginalReturn(size=(N, N_t-1),
                                                        dt=dt, seed=seed)
        returnsPathsArray[:, 1:] = marginalReturnsArray.cumsum(axis=1)
        return returnsPathsArray

    def __repr__(self):
        """
        Return a string representation of the VolBasket object.

        Returns
        -------
        str
            The string representation of the VolBasket object.
        """
        return f"VolBasket({dict(self.__assets)})"

    @property
    def assets(self) -> dict[VolIndex]:
        """
        Retrieve the volatility indices included in the MultiVolIndex object.

        Returns
        -------
        dict[VolIndex]
            A dictionary containing the volatility indices,
            where the keys represent the names of the indices
            and the values represent the corresponding VolIndex objects.
        """
        return self.__assets
    
    @property
    def nbAssets(self) -> dict[VolIndex]:
        return self.__nbAssets

    @property
    def corr(self) -> np.ndarray:
        """
        Retrieve the correlation matrix between the volatility indices.

        Returns
        -------
        np.ndarray[np.floating]
            The correlation matrix between the volatility indices.
        """
        return self.__corr

    @property
    def cov(self) -> np.ndarray:
        """
        Retrieve the covariance matrix between the volatility indices.

        Returns
        -------
        np.ndarray[np.floating]
            The covariance matrix between the volatility indices.
        """
        vols_diagMat = np.diag(self.__vols)
        return vols_diagMat @ self.__corr @ vols_diagMat

    @property
    def drift(self) -> np.ndarray:
        """
        Retrieve the drifts of the volatility indices.

        Returns
        -------
        np.ndarray
            An array of drift values for each volatility index.
        """
        return self.__drifts
