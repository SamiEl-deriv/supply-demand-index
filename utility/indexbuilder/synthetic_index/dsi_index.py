from stochastic_process_base import StochasticProcess
import numpy as np
from typing import Optional

class DriftSwitchIndex(StochasticProcess):
    
    def __init__(self, drift: float, volatility: float, gamma: float, regime_duration: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__mu = drift
        self.__sigma = volatility
        self.__gamma = gamma
        self.__T = regime_duration
        self.__current_state = 0     # assumption - we begin from state = 0 (can possibly be 1 or 2 as well)

        # Transition matrix
        self.__transition_matrix = np.array([
            [1 - 1/self.__T , self.__gamma * 1/self.__T , 1/(2*self.__T)],
            [1/(2*self.__T), 1 - 1/self.__T , 1/(2*self.__T)],
            [1/(2*self.__T), (1-self.__gamma) * (1/self.__T) , 1 - 1/self.__T]
        ])

    @property
    def drift(self):
        return self.__mu
    @property
    def volatility(self):
        return self.__sigma
    @property
    def drift_correction(self):
        return self.__gamma
    @property
    def regime_duration(self):
        return self.__T
    @property
    def current_state(self):
        return self.__current_state
    @property
    def transition_matrix(self):
        return self.__transition_matrix

    def calc_next_markov_state(self):
        transition_probabilities = np.transpose(self.__transition_matrix)[self.__current_state]
        next_state = np.random.choice([0, 1, 2], p=transition_probabilities)  # choose new state , based on probability of transitioning
        return next_state

    def calc_drift_func(self):
        self.__current_state = self.calc_next_markov_state()  # update old state to new updated state 
        drift_map = {0: self.__mu,         # find drift value based on current state
        1: 0, 
        2: -1*self.__mu}
        return drift_map.get(self.__current_state, None)

    def new_return(self, seed: Optional[int] = None) -> float:
        drift = self.calc_drift_func()
        stochastic_term = self.__sigma * np.sqrt(self.dt) * np.random.normal(loc=0, scale=1)
        return_val = (drift - 0.5 * self.__sigma ** 2) * self.dt + stochastic_term
        return return_val
