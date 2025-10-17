from stochastic_process_base import StochasticProcess
import numpy as np
from typing import Optional


class VolatilityIndex(StochasticProcess):
    def __init__(self, volatility: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__volatility = volatility
    
    @property
    def volatility(self):
        return self.__volatility
    
    def new_return(self, seed: Optional[int] = None) -> float:
        return (-self.__volatility ** 2 / 2) * self.dt + self.__volatility * np.random.normal(0, np.sqrt(self.dt))
