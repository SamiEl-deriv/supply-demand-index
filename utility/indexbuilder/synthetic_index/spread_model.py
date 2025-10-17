import numpy as np
from scipy.stats import norm

def PercentSpread(self, percentage: float = 0.05):
    spread = self._StochasticProcess__index[-1] * percentage / 100
    bid = self._StochasticProcess__index[-1] - spread/2
    ask = self._StochasticProcess__index[-1] + spread/2
    self._StochasticProcess__ask.append(bid)
    self._StochasticProcess__bid.append(ask)
    self._StochasticProcess__spread.append(spread)


def VSISpread(self):
    if len(self.index) == 1 :
        state_vect = np.zeros(shape = (self._VolatilitySwitchIndex__dim))
        state_vect[0] = 1
        self._StochasticProcess__ask.append(self.index[-1])
        self._StochasticProcess__bid.append(self.index[-1])
        self._StochasticProcess__spread.append(0)
    else :
        log_return = np.log(self.index[-1]) - np.log(self.index[-2])
        density_vect = norm.pdf(log_return, -(self.volatility*self._StochasticProcess__dt)/2, self.volatility*(self._StochasticProcess__dt)**(0.5))
        # print(density_vect)
        state_vect = self.transition_prob_matrix.dot(density_vect * self.prev_state / density_vect.dot(self.prev_state))
        spread =  self.index[-1] * self.volatility.dot(state_vect)*(self._StochasticProcess__dt)**(0.5)
        # print('spread', spread)
        bid = self.index[-1] - spread /2 
        ask = self.index[-1] + spread /2
        self._StochasticProcess__ask.append(bid)
        self._StochasticProcess__bid.append(ask)
        self._StochasticProcess__spread.append(spread)
        # state_vect = 2 
    self.prev_state = state_vect
    # print(self.prev_state)

def NoSpread(self):
    self._StochasticProcess__ask.append(self.index[-1])
    self._StochasticProcess__bid.append(self.index[-1])


spread_dict = {'default': PercentSpread,
               'percentage': PercentSpread,
               'no spread': NoSpread,
               'vsi_default': VSISpread}