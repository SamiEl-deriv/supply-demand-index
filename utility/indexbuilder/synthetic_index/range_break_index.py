from stochastic_process_base import StochasticProcess
import numpy as np
from scipy.stats import norm
from math import pi, erf, exp
from typing import Optional

class RangeBreakIndex(StochasticProcess):
    def __init__(self, step_size: float, perc_out: float, jump_param: float, wait_time: int, **kwargs):
        super().__init__(**kwargs)
    
        self.step_size = step_size
        self.perc_out = perc_out
        self.mean_jump = step_size * (1 - perc_out) / perc_out
        self.jump_param = jump_param
        self.wait_time = wait_time
        self.last_break_is_jump = False
        self.border_adjusted = False
        self.U = self.start_val
        self.L = self.start_val
        self.S_min = self.start_val
        self.S_max = self.start_val
        self.mu = 1
        self.sigma = 1

    def new_return(self, seed: Optional[int] = None) -> float:
        X1 = np.random.uniform(0, 1)
        if self.U == self.L or self.U == self.S_max or self.L == self.S_min:
            direction = 1 if self.U == self.S_max else -1
            if X1 <= self.perc_out:
                jump = round(self.mean_jump * (self.folded_norm(1, 1) / self.folded_mean + self.jump_param) / (1 + self.jump_param))
                next_val = direction * jump
                self.U = next_val if direction == 1 else self.U
                self.L = self.U if direction == 1 else next_val
                self.S_min = next_val
                self.S_max = next_val
                self.last_break_is_jump = direction == 1
                self.border_adjusted = False
            else:
                next_val = -direction * self.step_size
                self.S_min = min(next_val, self.S_min)
                self.S_max = max(next_val, self.S_max)
                self.border_adjusted = True
        else:
            direction = 1 if X1 < 0.5 else -1
            next_val = direction * self.step_size
            self.S_min = min(next_val, self.S_min)
            self.S_max = max(next_val, self.S_max)
            self.border_adjusted = True
        return next_val

    def folded_norm(self, m: float = 0, s: float = 1) -> float:
        return abs(np.random.normal(m, s))

    @property
    def folded_mean(self) -> float:
        return self.sigma * np.sqrt(2 / np.pi) * np.exp(-self.mu ** 2 / (2 * self.sigma ** 2)) + self.mu * (1 - 2 * norm.cdf(-self.mu / self.sigma))
