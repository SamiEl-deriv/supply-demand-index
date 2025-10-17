from stochastic_process_base import StochasticProcess
import numpy as np
from typing import Optional


class BearBullIndex(StochasticProcess):
    # St = St-1 * exp{ [-dividend_rate -sigma**2/2] * dt + sigma * sqrt(dt)*X } , where X is sample from a Normal Dist N(0,1). ----> sqrt(dt) * N(0,1) ~ N(0,dt) , where 0=Mean & dt=Variance
    # St = St-1 * exp{ [-dividend_rate -sigma**2/2] * dt + sigma * N(0,dt) } ---> final formula
    def __init__(self, dividend_rate: float, volatility:float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dividend_rate = dividend_rate  # bear = 20 ; bull = -35
        self.__sigma = volatility
    
    @property
    def dividend_rate(self):
        return self.__dividend_rate
    @property
    def volatility(self):
        return self.__sigma
    
    def new_return(self, seed: Optional[int] = None) -> float:
        return (-self.__dividend_rate - self.__sigma ** 2 / 2) * self.dt + self.__sigma * np.random.normal(0, np.sqrt(self.dt))
