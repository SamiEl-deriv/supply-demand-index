from stochastic_process_base import StochasticProcess
import numpy as np
from typing import Optional
from scipy.stats import poisson


class DexIndex(StochasticProcess):
    def __init__(self,
                 interest_rate: float,
                 volatility: float,
                 jump_frq: int,
                 proba_jump_up: float,
                 jump_up_size: float,
                 jump_down_size: float,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.__r = interest_rate
        self.__sigma = volatility
        self.__lambda = jump_frq
        self.__pjp = proba_jump_up
        self.__njp = 1 - proba_jump_up
        self.__pjs = jump_up_size    
        self.__njs = jump_down_size
        self.__rv = poisson(mu=self.__lambda * self.dt)
    
    def dex_jump_dist(self):
        x = np.random.uniform(0, 1) 
        if x < self.__njp:
            j = self.__njs * np.log(x/self.__njp)
        elif x >= self.__njp:
            j = -self.__pjs * np.log((1-x)/self.__pjp) 
        return j
    
    def random_poisson_value(self):
        random_poisson_number = self.__rv.rvs()
        return random_poisson_number
    
    def alpha(self):
        # from moment generation function
        result = self.__njp / (self.__njs + 1) + self.__pjp / (1 - self.__pjs) - 1
        return result
    
    def new_return(self, seed: Optional[int] = None):
        return_val = (self.__r-self.__sigma**2/2-self.__lambda*self.alpha())*self.dt + self.__sigma*np.random.normal(0, np.sqrt(self.dt)) + np.sum(self.dex_jump_dist() for i in range(0, self.__rv.rvs()))
        return return_val
