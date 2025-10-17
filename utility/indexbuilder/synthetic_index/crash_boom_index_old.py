from stochastic_process_base import StochasticProcess#, arithmeticProcess, dt
from scipy.optimize import fsolve 
from scipy.stats import norm, foldnorm, bernoulli
from numpy.random import SeedSequence
from typing import Type, Optional
import numpy as np


class CrashIndex(StochasticProcess):

    def __init__(self, start_val: float, mean_interval: int, dt: float = 1/(365 * 24 * 3600)) -> None:

        # mean_interval : The mean duration of the interval between two crashes
        global mu
        mu = np.sqrt(2 / np.pi) * np.exp(-1 / 2) + 1 - 2 * norm.cdf(-1)
        super().__init__(dt=dt,start_val=start_val)
        self.__Pu, self.__Pd, self.__MUT, self.__MDT, \
            self.__volatility = CrashIndex.get_params(self, mean_interval)
         
    @staticmethod
    def get_params(self, mean_interval: int) -> tuple[float, float, float,
                                                float, float]:

        P_d = 1 / mean_interval
        P_u = 1 - P_d
        MDT = np.log(P_u) / np.sqrt(self.dt)
        t2 = MDT * self.dt / mu
        def factor(x: float): return np.exp(x**2 / 2 + x) * \
            norm.cdf(1 + x) + np.exp(x**2 / 2 - x) * norm.cdf(x - 1)

        def f(x: float): return 1 - factor(t2) * P_d - factor(x) * P_u
        z = fsolve(f, 0)[0]
        MUT = z / self.dt * mu
        VOL = np.sqrt(2 / mu**2 * (P_u * MUT**2 + P_d * MDT**2 +
                      4 * P_d * P_u * MUT * MDT) - (P_u * MUT + P_d * MDT)**2)
        return P_u, P_d, MUT, MDT, VOL

    @property
    def crash_prob(self) -> float:

        return self.__Pu

    @property
    def normal_prob(self) -> float:

        return self.__Pd

    @property
    def MUT(self) -> float:

        return self.__MUT

    @property
    def MDT(self) -> float:

        return self.__MDT

    @property
    def volatility(self) -> float:

        return self.__volatility
    
    def new_return(self, seed: Optional[int] = None) -> float:
        rand_norm = foldnorm.rvs(1, random_state=seed)

        seed2 = SeedSequence(seed).spawn(1)[0].generate_state(1)[0]
        rand_upOrdDown = bernoulli.rvs(self.__Pu, size=1,
                                       random_state=seed2) \
            * (self.__MUT - self.__MDT) + self.__MDT
        return (rand_upOrdDown * rand_norm / mu * np.sqrt(self.dt))[0]


class BoomIndex(StochasticProcess):

    def __init__(self,start_val: float, mean_interval: int, dt: float) -> None:

        global mu
        mu = np.sqrt(2 / np.pi) * np.exp(-1 / 2) + 1 - 2 * norm.cdf(-1)

        super().__init__(dt=dt,start_val=start_val)
        self.__Pu, self.__Pd, self.__MUT, self.__MDT, \
            self.__volatility = BoomIndex.get_params(self,mean_interval)

    @staticmethod
    def get_params(self,mean_interval: int) -> tuple[float, float, float,
                                                float, float]:

        P_u = 1 / mean_interval
        P_d = 1 - P_u
        MUT = np.log(P_u) / np.sqrt(self.dt)
        t2 = MUT * self.dt / mu
        def factor(x: float): return np.exp(x**2 / 2 + x) * \
            norm.cdf(1 + x) + np.exp(x**2 / 2 - x) * norm.cdf(x - 1)

        def f(x: float): return 1 - factor(t2) * P_d - factor(x) * P_u
        z = fsolve(f, 0)[0]
        MDT = z / self.dt * mu
        VOL = np.sqrt(2 / mu**2 * (P_u * MUT**2 + P_d * MDT**2 +
                      4 * P_d * P_u * MUT * MDT) - (P_u * MUT + P_d * MDT)**2)
        return P_u, P_d, MUT, MDT, VOL

    @property
    def boom_prob(self) -> float:

        return self.__Pu

    @property
    def normal_prob(self) -> float:

        return self.__Pd

    @property
    def MUT(self) -> float:

        return self.__MUT

    @property
    def MDT(self) -> float:

        return self.__MDT

    @property
    def volatility(self) -> float:

        return self.__volatility

    def new_return(self, seed: Optional[int] = None) -> float:
        rand_norm = foldnorm.rvs(1, random_state=seed)

        seed2 = SeedSequence(seed).spawn(1)[0].generate_state(1)[0]
        rand_upOrDown = bernoulli.rvs(self.__Pu, size=1,
                                       random_state=seed2) \
            * (self.__MUT - self.__MDT) + self.__MDT
        return rand_upOrDown * rand_norm / mu * np.sqrt(self.dt)
