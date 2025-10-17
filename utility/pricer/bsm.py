import numpy as np
from scipy.stats import norm
from deriv_quant_package.pricer.greeks import Greeks


class BSM:
    """
    A class used to value vanilla call and put options using the Black-Scholes-Merton model given a set of parameters and when

    Attributes
    ----------
    r : float
        The interest rate (FX: domestic rate)
    q : float
        The dividend rate (FX: foreign rate)
    market_quote : dict[str : str | dict[float : float]] | None
            The forward delta or strike here must be unpacked before use to price a contract, i.e. no spot delta or premium adjusted delta

            Format:
            - {"convention" = "delta" | "strike" | "moneyness", "values": {x1 : mkt_vol1, .... xN : mkt_volN}}
            - delta must be in Call convention like 25C 50C 75C
            - values expects a dict[int | float : float]

            To be extended as necessary in the future
    T : float
        The term to maturity/tenor in day fractions of a year
    sign : {1,-1}
        1 if call or -1 if put
    put : bool
        The class values a put option if true and a call option if false
    """

    # Minimum step used for finite differences is set as the cube root of the machine epsilon
    # (upper bound on the relative approximation error due to rounding)
    # Typically applicable for first order central differences, but we'll be using it for
    # second order differences too
    # Sauer, Timothy (2012). Numerical Analysis. Pearson. p.248.
    MIN_STEP = np.power(np.finfo(float).eps, 1 / 3)

    def __init__(self, r, T, q=0, market_quote=None, put=False) -> None:
        """
        Parameters
        ----------
        r : float
            The interest rate (FX: domestic rate)
        T : float
            The term to maturity/tenor
        q : float (DEFAULT = 0)
            The dividend rate (FX: foreign rate)
        market_quote : dict[str : str | dict[float : float]] | None (DEFAULT = None)
            The forward delta or strike here must be unpacked before use to price a contract, i.e. no spot delta or premium adjusted delta

            Format:
            - {"convention" = "delta" | "strike" | "moneyness", "values": {x1 : mkt_vol1, .... xN : mkt_volN}}
            - delta must be in Call convention like 25C 50C 75C
            - values expects a dict[int | float : float]
        put : bool (DEFAULT = False)
            If true, option is a put, otherwise a call

        """

        self.r = r
        self.q = q
        self.T = T
        self.put = put
        self.sign = -1 if self.put else 1
        self.greeks = Greeks(r, q, T, put)
        self.market_quote = market_quote

        if self.market_quote is not None:
            convention = self.market_quote["convention"]

            # Handle input convention
            if convention not in ["delta", "strike", "moneyness"]:
                raise TypeError(
                    "Wrong convention given, only delta, moneyness or strike is acceptable")

            self.sort_vol_strike = self.sort_vol_key if (
                convention == "strike") else None

            self.sort_vol_key = [str(j) for j in sorted(
                [int(i) for i in self.market_quote["values"].keys()], reverse=True)]
            self.market_quote["values"] = {
                str(int(k)): v for k, v in self.market_quote["values"].items()}

            self.sort_vol = [self.market_quote["values"][i]
                             for i in self.sort_vol_key]
            self.sort_vol_strike = self.sort_vol_key if (
                convention == "strike") else None

    def __call__(self, S, K, vol) -> float:
        """
        Wrapper for get_price

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
            BS price
        """

        return self.get_price(S, K, vol)

    def __str__(self) -> str:
        """
        Prints Black scholes pricer details

        Returns
        -------
        str
            BS pricer details - put/call, r ,q ,market_vol, T
        """

        string = type(self).__name__ + " " + \
            ("put" if self.put else "call") + " option pricer with\n"
        string += f"r={self.r}\n" + f"q={self.q}\n" + \
            f"market_quote={self.market_quote}\n" + f"T={self.T}"
        return string

    def get_price(self, S, K, vol=None) -> float:
        """
        Calculates the Black-Scholes call price according the following formulas

        C = N(d1)*S*e^(-q*T) - N(d2)*K*e^(-r*T)

        P = N(-d2)*K*e^(-r*T) - N(-d1)*S*e^(-q*T)

        Parameters
        ----------
        S : float
            The spot price
        vol : float (DEFAULT = None)
            The volatility
        K : float
            The strike price, default: S

        Returns
        -------
        float
            BS price
        """

        # Cast strikes & spots to an np array
        if (isinstance(K, list)) | (isinstance(K, tuple)):
            K = np.array(K)

        # Volatility is required to price a contract
        if vol is None:
            raise Exception("Please enter volatility to price a contract")

        s = -1 if self.put else 1
        d1 = self.greeks.d1(S=S, K=K, vol=vol)
        d2 = self.greeks.d2(S=S, K=K, vol=vol)

        return s * (norm.cdf(s * d1) * S * np.exp(-self.q * self.T) -
                    norm.cdf(s * d2) * K * np.exp(-self.r * self.T))

    def delta(self, S, K, vol=None, forward=False, fd=False) -> float:
        """
        Calculates the spot/forward delta of the option,
        i.e the first derivative w.r.t the spot or forward

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        forward : bool (DEFAULT = False)
            evaluate forward delta if true, else spot delta
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The spot/forward delta of the option
        """
        if fd:  # Use central differences

            dS = self.MIN_STEP

            C1 = self.formula(S - dS, K, self.r, self.q,
                              self.vol, self.T, self.put)
            C2 = self.formula(S + dS, K, self.r, self.q,
                              self.vol, self.T, self.put)

            forward_adj = np.exp(self.q * self.T) if forward else 1
            return forward_adj * (C2 - C1) / dS

        return self.greeks.delta(S, vol, K, forward)

    def dual_delta(self, S, vol, K=None, fd=False) -> float:
        """
        Calculates the dual delta of the option,
        i.e the first derivative w.r.t the strike

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The dual delta of the option
        """

        if fd:  # Use central differences

            dK = self.MIN_STEP

            C1 = self.formula(S, K - dK, self.r, self.q,
                              self.vol, self.T, self.put)
            C2 = self.formula(S, K + dK, self.r, self.q,
                              self.vol, self.T, self.put)

            return (C2 - C1) / dK

        return self.greeks.dual_delta(S, vol, K)

    def delta_pa(self, S, vol, K=None, forward=False) -> float:
        """
        Calculates the premium-adjusted spot/forward delta of the option,
        i.e the first derivative w.r.t the spot/forward, adjusted with the premium

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        forward : bool (DEFAULT = False)
            evaluate forward delta if true, else spot delta
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The premium-adjusted spot/forward put/call delta of the option
        """
        return self.greeks.delta_pa(S, vol, K, forward)

    def vega(self, S, vol, K=None, fd=False) -> float:
        """
        Calculates the vega of the option,
        i.e the derivative of the option price w.r.t volatility

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The vega of the option
        """

        if fd:  # Use central differences

            dvol = self.MIN_STEP

            C1 = self.formula(S, K, self.r, self.q,
                              self.vol - dvol, self.T, self.put)
            C2 = self.formula(S, K, self.r, self.q,
                              self.vol + dvol, self.T, self.put)

            return (C2 - C1) / dvol

        return self.greeks.vega(S, vol, K)

    def vanna(self, S, vol, K=None, fd=False) -> float:
        """
        Calculates the vanna of the option,
        i.e the mixed second derivative of the option price w.r.t volatility and spot
        Parameters

        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The vanna of the option
        """

        if fd:  # Use central differences

            dS = dvol = self.MIN_STEP

            C1 = self.formula(S - dS, K, self.r, self.q,
                              self.vol - dvol, self.T, self.put)
            C2 = self.formula(S + dS, K, self.r, self.q,
                              self.vol + dvol, self.T, self.put)
            C3 = self.formula(S + dS, K, self.r, self.q,
                              self.vol - dvol, self.T, self.put)
            C4 = self.formula(S - dS, K, self.r, self.q,
                              self.vol + dvol, self.T, self.put)

            return (C1 + C2 - C3 - C4) / (4 * dvol * dS)

        return self.greeks.vanna(S, vol, K)

    def volga(self, S, vol, K, fd=False) -> float:
        """
        Calculates the volga of the option,
        i.e the second derivative of the option price w.r.t volatility

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)
        fd : bool (DEFAULT = False)
            Returns approximated derivative if True

        Returns
        -------
        float
            The volga of the option
        """

        if fd:  # Use central differences

            dvol = self.MIN_STEP

            C1 = self.formula(S, K, self.r, self.q,
                              self.vol - dvol, self.T, self.put)
            C2 = self.formula(S, K, self.r, self.q, self.vol, self.T, self.put)
            C3 = self.formula(S, K, self.r, self.q,
                              self.vol + dvol, self.T, self.put)

            return (C1 - 2 * C2 + C3) / (dvol ** 2)

        return self.greeks.volga(S, vol, K)

    @classmethod
    def formula(
            S: float,
            K: float,
            r: float,
            q: float,
            vol: float,
            T: float,
            put: bool):
        """
        Raw formula for testing & FD Greek purposes
        """
        d1 = (np.log(S / K) + (r - q + vol**2 / 2)) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        s = -1 if put else 1

        return s * (S * np.exp(-q * T) * norm.cdf(s * d1) -
                    K * np.exp(-r * T) * norm.cdf(s * d2))


if __name__ == "__main__":
    '''
    r = 0.03
    t = 1/12
    local_pricer = BSM(r=r, T=T, q=0, put=False)
    S = 100
    K = 100
    vol = 0.2
    local_pricer.get_price(S=S, K=K, vol=vol)
    '''
