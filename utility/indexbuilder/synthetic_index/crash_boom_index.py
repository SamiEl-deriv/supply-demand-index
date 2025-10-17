from stochastic_process_base import StochasticProcess#, arithmeticProcess, dt
from scipy.optimize import fsolve 
from scipy.stats import norm, foldnorm, bernoulli, invgauss
from numpy.random import SeedSequence
from typing import Type, Optional
import numpy as np
from math import pi, erf, exp, log

class CrashIndex(StochasticProcess):
    def __init__(self,
                 mean_interval: float,
                 diff_percent: float,
                 mdt: float,
                 **kwargs) -> None:
        # mean_interval : The mean duration of the interval between two crashes
        super().__init__(**kwargs)
        self.__mdt = mdt
        self.__diff_percent = diff_percent   # percentage difference between adjacent ticks --> influences Falls ticks and normal ticks
        self.__p_up = 1 / mean_interval
        self.__p_down = 1 - self.__p_up 
        self.__mut = -self.__mdt*self.__p_up/self.__p_down
        self.__mu = 1
        self.__sigma = 1
    
    def long_term_mean(self):
        # Long Term Mean of folded normal with parameters mu & sigma
        mu_y = self.__sigma * np.sqrt(2/pi) * exp(-self.__mu**2/(2*self.__sigma**2)) + self.__mu * (1-2*norm.cdf(-self.__mu/self.__sigma))
        return mu_y

    @property
    def crash_prob(self) -> float:
        return self.__p_down
    @property
    def normal_prob(self) -> float:
        return self.__p_up
    @property
    def MUT(self) -> float:
        return self.__mut 
    @property
    def MDT(self) -> float:
        return self.__mdt 

    def new_return(self, seed: Optional[int] = None) -> float:
        self.__mu_y = self.long_term_mean()
        rand_uniform = np.random.rand() 
        # folded normal distribution
        sample = np.random.normal(self.__mu, self.__sigma)
        folded_sample = abs(sample)
        # final value
        return np.where(rand_uniform > self.__p_up, self.__mut, self.__mdt) * folded_sample * np.sqrt(self.dt) / self.__mu_y 


class BoomIndex(StochasticProcess):
    def __init__(self,
                 mean_interval: float,
                 diff_percent: float,
                 mut: float,
                 **kwargs) -> None:
        # mean_interval : The mean duration of the interval between two crashes
        super().__init__(**kwargs)
        self.__diff_percent = diff_percent   # percentage difference between adjacent ticks --> influences Jump ticks and normal ticks
        self.__mut = mut
        self.__p_down = 1 / mean_interval
        self.__p_up = 1 - self.__p_down
        self.__mdt = -self.__mut * self.__p_down / self.__p_up
        self.__mu = 1
        self.__sigma = 1

    def long_term_mean(self):
        # Long Term Mean of folded normal with parameters mu & sigma
        mu_y = self.__sigma * np.sqrt(2/pi) * exp(-self.__mu**2/(2*self.__sigma**2)) + self.__mu * (1-2*norm.cdf(-self.__mu/self.__sigma))
        return mu_y

    @property
    def boom_prob(self) -> float:
        return self.__p_up
    @property
    def normal_prob(self) -> float:
        return self.__p_down
    @property
    def MUT(self) -> float:
        return self.__mut 
    @property
    def MDT(self) -> float:
        return self.__mdt 
    
    def new_return(self, seed: Optional[int] = None) -> float:
        self.__mu_y = self.long_term_mean()
        rand_uniform = np.random.rand() 
        # folded normal distribution
        sample = np.random.normal(self.__mu, self.__sigma)
        folded_sample = abs(sample)
        # final value
        return np.where(rand_uniform > self.__p_up, self.__mut, self.__mdt) * folded_sample * np.sqrt(self.dt) / self.__mu_y 
