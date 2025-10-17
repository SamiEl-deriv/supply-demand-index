import numpy as np
from scipy.stats import norm
from deriv_quant_package.pricer.bsm import BSM
# from ...tests import test_pricer
import scipy.optimize as so


class VV(BSM):
    """
    A class used to value vanilla call and put options using the Vanna-Volga method given set of parameters and when

    Attributes
    ----------
    r : float
        The interest rate (FX: domestic rate)
    q : float
        The dividend rate (FX: foreign rate)
    market_data : dict
        The forward delta or strike here must be unpacked before use to price a contract, i.e. no spot delta or premium adjusted delta

        - delta must be in Call convention like 25C 50C 75C
        - expect a float or integer as a keys and value of market quote
            (delta1 (int + string) or strike1 (float + string) : mktvol1 <-  integer or float
        - expect 3 key and 3 value only (can be extend to more in future)

        dict {"convention" : "delta" | "strike", "values" : {delta1 : mkt_vol1, .... deltaN : mkt_volN}}
    t : float
        The term to maturity/tenor in day fractions of a year
    sign : {1,-1}
        1 if call or -1 if put
    put : bool
        The class values a put option if true and a call option if false
    """

    def __init__(self, r, T, market_data, q=0, put=False) -> None:
        """
        Parameters
        ----------
        K : float
            The strike price
        r : float
            The interest rate (FX: domestic rate)
        q : float
            The dividend rate (FX: foreign rate)
        market_data : dict - {"convention" = "delta" | "strike" | "moneyness", "values": {x1 : mkt_vol1, .... xN : mkt_volN}}
            The forward delta or strike here must be unpacked before use to price a contract, i.e. no spot delta or premium adjusted delta

            - delta must be in Call convention like 25C 50C 75C
            - expect a float or integer as a keys and value of market quote
                (delta1 (int + string) or strike1 (float + string) : mktvol1 <-  integer or float
            - expect 3 keys and 3 values only (can be extend to more in future)
        T : float
            The term to maturity/tenor
        """

        super().__init__(r=r, q=q, T=T, put=put, market_data=market_data)
        self.forward = None

    def forward_for_strike(self, S) -> None:
        """
        Sets forward and calibrate strikes if given
        """
        forward = S * np.exp((self.r - self.q) * self.T)
        self.sort_vol_strike = [forward * np.exp(
            -self.sign * norm.ppf(self.sign * float(self.sort_vol_key[i]) / 100) * self.sort_vol[i] * np.sqrt(
                self.T) + 0.5 * self.sort_vol[i] * self.sort_vol[i] * np.sqrt(self.T)) for i in
            range(len(self.sort_vol))]

    # def vega(self, S, vol=None, K=None) -> float:
    #     """
    #     Calculates the vega of the option,
    #     i.e the derivative of the option price w.r.t volatility
    #     Parameters
    #     ----------
    #     S : float
    #         The spot price
    #     K : float
    #         The strike price, default: S

    #     Returns
    #     -------
    #     float
    #         The vega of the option
    #     """

    #     if K is None:
    #         K = S

    #     if vol is None:
    #         raise Exception("Please enter volatility to value a greek")

    #     d1 = self._d1(S=S, K=K, vol=vol)

    #     return S * np.exp(-self.q * self.t) * np.sqrt(self.t) * norm.pdf(d1)

    def x_weight(self, S, K, label):
        # K can be any reasonable K we want to price
        # here we assume three mkt point for replicating portfolio
        bs_vol = self.sort_vol[len(self.sort_vol) // 2]
        saved_array = np.array([])
        local = np.any(np.isin(K, self.sort_vol_strike))
        saved = dict()
        mask = None
        if local:
            for i in range(len(self.sort_vol_strike)):
                saved[i] = np.where(K == self.sort_vol_strike[i])[0]
            # if K match market K then record the location
            for i in list(saved.keys()):
                saved_array = np.concatenate([saved_array, saved[i]])
        # else:
        #     pass

        if saved_array.size == 0:
            adj_quoted_K = K
        else:
            mask = np.full(len(K), True)
            mask[saved_array] = False
            adj_quoted_K = K[mask]

        vk = self.greeks.vega(S=S, vol=bs_vol, K=adj_quoted_K)

        if label == 0:
            vk1 = self.greeks.vega(S=S, vol=bs_vol, K=self.sort_vol_strike[0])
            up_left = np.log(np.divide(self.sort_vol_strike[1], adj_quoted_K))
            up_right = np.log(np.divide(self.sort_vol_strike[2], adj_quoted_K))
            deno = np.log(
                self.sort_vol_strike[1] / self.sort_vol_strike[0]) * np.log(
                self.sort_vol_strike[2] / self.sort_vol_strike[0])
            based_result = np.multiply(np.divide(vk, vk1), np.divide(
                np.multiply(up_left, up_right), deno))

        elif label == 1:
            vk2 = self.greeks.vega(S=S, vol=bs_vol, K=self.sort_vol_strike[1])
            up_left = np.log(np.divide(adj_quoted_K, self.sort_vol_strike[0]))
            up_right = np.log(np.divide(self.sort_vol_strike[2], adj_quoted_K))
            deno = np.log(
                self.sort_vol_strike[1] / self.sort_vol_strike[0]) * np.log(
                self.sort_vol_strike[2] / self.sort_vol_strike[1])
            based_result = np.multiply(np.divide(vk, vk2), np.divide(
                np.multiply(up_left, up_right), deno))

        elif label == 2:
            vk3 = self.greeks.vega(S=S, vol=bs_vol, K=self.sort_vol_strike[2])
            up_left = np.log(np.divide(adj_quoted_K, self.sort_vol_strike[0]))
            up_right = np.log(np.divide(adj_quoted_K, self.sort_vol_strike[1]))
            deno = np.log(
                self.sort_vol_strike[2] / self.sort_vol_strike[0]) * np.log(
                self.sort_vol_strike[2] / self.sort_vol_strike[1])
            based_result = np.multiply(np.divide(vk, vk3), np.divide(
                np.multiply(up_left, up_right), deno))

        else:
            raise Exception("label error")
        if not local:
            return based_result
        else:
            # if K match "market_K" then return 1 at the matched strike and 0
            # for others
            lala = np.full(len(K), 0)
            lala[mask] = based_result
            if saved_array.size == 0:
                return lala
            else:
                lala[saved[label]] = 1
                return lala

    def get_price(self, S, K):
        """
        Calculates the Vanna-Volga call price according the following formulas

        Parameters
        ----------
        S : float
            The spot price
        vol : float
            The volatility
        K : float
            The strike price, default: S

        Returns
        -------
        float
            VV price
        """
        if self.market_data["convention"] == "delta":
            if self.forward is None:
                self.forward_for_strike(S)
            else:
                pass

        else:
            pass

        if (isinstance(K, list)) | (isinstance(K, tuple)) | (
                isinstance(K, np.ndarray)):
            if isinstance(K, np.ndarray):
                pass
            else:
                K = np.array(K)
            K = K.flatten()
            K_shape = K.shape[0]
        else:
            K_shape = 1

        # print(self.T, self.r, self.q, self.sort_vol[len(self.sort_vol) // 2 + 1], S, K)
        BS_price = super().get_price(
            S, K=K, vol=self.sort_vol[len(self.sort_vol) // 2 + 1])
        x_weight1 = np.zeros((K_shape, 3))
        y_diff = np.zeros((3, 1))
        for i in range(x_weight1.shape[1]):
            # weight to market option
            x_weight1[:, i] = self.x_weight(S, K, label=i)
            # vol adjustment
            y_diff[i,
                   0] = super().get_price(S=S,
                                          vol=self.sort_vol[i],
                                          K=self.sort_vol_strike[i]) - super().get_price(S=S,
                                                                                         vol=self.sort_vol[len(self.sort_vol) // 2 + 1],
                                                                                         K=self.sort_vol_strike[i])

        return BS_price + np.dot(x_weight1, y_diff).flatten()

    # def run_test(self):
    #     test_case = test_pricer.Test_Pricer(r=self.r, q=self.q)
    #     call_min_strike = self.get_price(S=test_case.S, K=test_case.very_small_K)
    #     call_max_strike = self.get_price(S=test_case.S, K=test_case.very_big_K)
    #     call_min_strike_add = self.get_price(S=test_case.S, K=test_case.very_small_K + test_case.very_small_K)
    #     call_max_strike_add = self.get_price(S=test_case.S, K=test_case.very_big_K + test_case.very_small_K)
    #     call_array = self.get_price(S=test_case.S, K=test_case.K)
    #     print("Start 5 Test Case")
    #     # Test pricer with K -> 0
    #     test_case.MinStrikeTest(call_min_strike)
    #     # Test pricer with K -> infinity
    #     test_case.MaxStrikeTest(call_max_strike)
    #     # Test first derivative of pricer respective to small K
    #     test_case.MinStrikeDerivativeTest(call_min_strike, call_min_strike_add)
    #     # Test first derivative of pricer respective to big K
    #     test_case.MaxStrikeDerivativeTest(call_max_strike, call_max_strike_add)
    #     # Test call function whether is convex
    #     test_case.ConvexTest(call_array)
    #     print("Test End")


if __name__ == "__main__":
    '''
    r = 0.03
    T = 1/12
    market_data = {"convention": "delta", "value": {"25": 0.3, "50": 0.2, "75": 0.25}}

    local_pricer = vannavolga.VV(r=r, T=T, market_data = market_data)
    local_pricer.get_vol(S = 100, K = np.arange(0, 2000, 1) / 10, kind="2")
    local_pricer.vol <--- this is vol for vanna volga
    '''
