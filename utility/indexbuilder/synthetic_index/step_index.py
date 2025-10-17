from stochastic_process_base import StochasticProcess
from stochastic_process_base import arithmeticProcess
from typing import Optional
import numpy as np


@arithmeticProcess
class StepIndex(StochasticProcess):
    def __init__(self, step_size: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__step_size = step_size

    @property
    def step_size(self):
        return self.__step_size

    def new_return(self, seed: Optional[int] = None) -> float:
        random = np.random.uniform(0, 1)
        step_val = self.__step_size if random > 0.5 else -self.__step_size
        return step_val
