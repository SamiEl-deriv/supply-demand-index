from stochastic_process_base import StochasticProcess
import numpy as np
from typing import Optional

class JumpIndex(StochasticProcess):
    # St = St-1 * exp{ [drift - sigma**2/2] * dt + sigma * sqrt(dt)*X1 + jump * sigma * sqrt(dt)*X2}  ;   sqrt(dt) * N(0,1) ~ N(0,dt) , replace in formuala
    # St = St-1 * exp{ [drift - sigma**2/2] * dt + sigma * N1(0,dt) + jump * sigma * N2(0,dt)}  ---> final formula
    def __init__(self, volatility: float, jump_factor: float = 30, drift: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__sigma = volatility
        self.__jump = jump_factor   # fix at 30
        self.__drift = drift

    @property
    def volatility(self):
        return self.__sigma

    @property
    def jump_factor(self):
        return self.__jump

    @property
    def drift(self):
        return self.__drift

    def new_return(self, seed: Optional[int] = None) -> float:
        if self.__drift is None:
            self.__drift = -1 * self.__jump**2 * self.__sigma**2 / 2
        return (self.__drift - self.__sigma**2 / 2) * self.dt + self.__sigma * np.random.normal(0, np.sqrt(self.dt)) + self.__jump * self.__sigma * np.random.normal(0, np.sqrt(self.dt))
