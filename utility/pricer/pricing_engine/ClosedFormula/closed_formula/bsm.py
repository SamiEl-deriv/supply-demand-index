from ..cf import ClosedFormula
from ....options.options import VanillaOption
import numpy as np
from scipy.stats import norm


class BSM(ClosedFormula):

    def __init__(self, Option: VanillaOption, spot: float):
        self.battery_type = battery_type
        super().__init__(Option, spot)

    def get_premium(self):
        # self.set_name()
        if self._name == 'PayOffDigitalCall':
            self._premium = norm.cdf(
                self._d2) * np.exp(-self._Option.riskFreeRate * self._Option.expiry)

            print('PayOffDigitalCall')
        else:
            print('bad payoff')
