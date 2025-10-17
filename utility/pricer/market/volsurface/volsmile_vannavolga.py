import numpy as np
from deriv_quant_package.pricer.vannavolga import VV
from .volsmile import VolSmile
# from deriv_quant_package.pricer.bsm import BSM
import scipy.optimize as so
import warnings

# TODO: Figure out multiple inheritance with VolSmile


class VolSmileVannaVolga(VV):
    def __init__(self, r, T, market_data, q=0, put=False) -> None:
        """
        A class used to value vanilla call and put options using the Vanna-Volga method given set of parameters and when

        TODO: To implement non-constant interet rates at any point?

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

            dict {"convention" = "delta" | "strike", "values": {delta1 : mkt_vol1, .... deltaN : mkt_volN}}
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        T : float
            The term to maturity/tenor in day fractions of a year
        put : bool
            The class values a put option if true and a call option if false
        """
        super().__init__(r=r, T=T, market_data=market_data, q=q, put=put)
        self.vol_type = None

    def __str__(self) -> str:
        """
        Prints Vanna-Volga volsmile details

        Returns
        -------
        str
            VV volsmile details - put/call, r ,q ,vol, T
        """

        string = f"Vanna-Volga volatility smile with\n"
        string += f"r={self.r}\n" + f"q={self.q}\n" + \
            f"market_data={self.market_data}\n" + f"T={self.T}"
        return string

    def first_approx(self, S, K):

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
        else:
            pass

        front = np.divide(
            np.multiply(
                np.log(
                    self.sort_vol_strike[1] / K),
                np.log(
                    self.sort_vol_strike[2] / K)),
            np.multiply(
                np.log(
                    self.sort_vol_strike[1] / self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] / self.sort_vol_strike[0]))) * self.sort_vol[0]
        mid = np.divide(
            np.multiply(
                np.log(
                    K / self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] / K)),
            np.multiply(
                np.log(
                    self.sort_vol_strike[1] / self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] / self.sort_vol_strike[1]))) * self.sort_vol[1]
        end = np.divide(
            np.multiply(
                np.log(
                    K / self.sort_vol_strike[0]),
                np.log(
                    K / self.sort_vol_strike[1])),
            np.multiply(
                np.log(
                    self.sort_vol_strike[2] / self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] / self.sort_vol_strike[1]))) * self.sort_vol[2]
        return front + mid + end

    def second_approx(self, S, K):

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
        else:
            pass

        big_D1 = self.first_approx(S, K) - self.sort_vol[1]

        big_D2_front = np.divide(
            np.multiply(
                np.log(
                    self.sort_vol_strike[1] /
                    K),
                np.log(
                    self.sort_vol_strike[2] /
                    K)),
            np.multiply(
                np.log(
                    self.sort_vol_strike[1] /
                    self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] /
                    self.sort_vol_strike[0])))
        d1_K1 = self.greeks.d1(
            S=S, vol=self.sort_vol[1], K=self.sort_vol_strike[0])
        d2_K1 = self.greeks.d2(
            S=S, vol=self.sort_vol[1], K=self.sort_vol_strike[0])
        var_diff_1 = (self.sort_vol[0] - self.sort_vol[1]) * \
            (self.sort_vol[0] - self.sort_vol[1])

        big_D2_end = np.divide(
            np.multiply(
                np.log(
                    K /
                    self.sort_vol_strike[0]),
                np.log(
                    K /
                    self.sort_vol_strike[1])),
            np.multiply(
                np.log(
                    self.sort_vol_strike[2] /
                    self.sort_vol_strike[0]),
                np.log(
                    self.sort_vol_strike[2] /
                    self.sort_vol_strike[1])))
        d1_K3 = self.greeks.d1(
            S=S, vol=self.sort_vol[1], K=self.sort_vol_strike[2])
        d2_K3 = self.greeks.d2(
            S=S, vol=self.sort_vol[1], K=self.sort_vol_strike[2])
        var_diff_3 = (self.sort_vol[2] - self.sort_vol[1]) * \
            (self.sort_vol[2] - self.sort_vol[1])

        big_D2 = np.multiply(
            np.multiply(
                big_D2_front,
                d1_K1),
            d2_K1) * var_diff_1 + np.multiply(
            np.multiply(
                big_D2_end,
                d1_K3),
            d2_K3) * var_diff_3

        d1d2_K = np.multiply(
            self.greeks.d1(
                S=S, vol=self.sort_vol[1], K=K), self.greeks.d2(
                S=S, vol=self.sort_vol[1], K=K))
        content = self.sort_vol[1] * self.sort_vol[1] + \
            np.multiply(d1d2_K, 2 * self.sort_vol[1] * big_D1 + big_D2)
        dan_loc = np.where(content < 0)[0]

        if dan_loc.shape[0] > 0:
            warnings.warn("Encounter negative square root")
        else:
            pass

        up = -self.sort_vol[1] + np.sqrt(content)
        return self.sort_vol[1] + np.divide(up, d1d2_K)

    def __call__(self, S, K, kind='vv') -> float:
        return self.get_vol(S, K, kind)

    def get_vol(self, S, K, kind='vv'):
        if kind == "vv":
            if (isinstance(K, np.ndarray)) | (
                    isinstance(K, list)) | (isinstance(K, tuple)):
                if isinstance(K, list):
                    K = np.array(K)
                elif isinstance(K, tuple):
                    K = np.array(K)
                else:
                    pass
                K = K.flatten()
            else:
                K = np.array([K])
            array_vol = np.zeros(K.shape[0])
            trigger = 0 if K.shape[0] == 1 else K.shape[0] // 2
            vv_price = self.get_price(S=S, K=K)
            loc = 0
            for i in range(K.shape[0]):
                if i < trigger:
                    if i == 0:
                        loc = trigger
                        res = so.basinhopping(
                            lambda x: abs(
                                vv_price[loc] -
                                super(
                                    VV,
                                    self).get_price(
                                    S=S,
                                    K=K[loc],
                                    vol=x)),
                            x0=self.sort_vol[1],
                            niter_success=5,
                            minimizer_kwargs={
                                "bounds": (
                                    (0.01,
                                     0.99),
                                )})
                        array_vol[loc] = res.x[0]
                    else:
                        loc += 1
                        res = so.basinhopping(lambda x: abs(vv_price[loc] - super(VV, self).get_price(S=S, K=K[loc], vol=x)),
                                              x0=array_vol[loc - 1], niter_success=5, minimizer_kwargs={"bounds": ((0.01, 0.99),)})
                        array_vol[loc] = res.x[0]
                else:
                    if i == trigger:
                        loc = trigger - 1
                        res = so.basinhopping(
                            lambda x: abs(
                                vv_price[loc] -
                                super(
                                    VV,
                                    self).get_price(
                                    S=S,
                                    K=K[loc],
                                    vol=x)),
                            x0=self.sort_vol[1],
                            niter_success=5,
                            minimizer_kwargs={
                                "bounds": (
                                    (0.01,
                                     0.99),
                                )})
                        array_vol[loc] = res.x[0]
                    else:
                        loc -= 1
                        res = so.basinhopping(lambda x: abs(vv_price[loc] - super(VV, self).get_price(S=S, K=K[loc], vol=x)),
                                              x0=array_vol[loc + 1], niter_success=5, minimizer_kwargs={"bounds": ((0.01, 0.99),)})
                        array_vol[loc] = res.x[0]
            self.vol = array_vol
            self.vol_type = "VV replication implied vol"

        elif kind == "1":
            self.vol = self.first_approx(S=S, K=K)
            self.vol_type = "first order approx"

        elif kind == "2":
            self.vol = self.second_approx(S=S, K=K)
            self.vol_type = "second order approx"

        else:
            raise ValueError("Please select kind = 'vv' | '1' |'2' only")

        return self.vol[0] if isinstance(
            self.vol, (list, np.ndarray)) and len(
            self.vol) == 1 else self.vol
