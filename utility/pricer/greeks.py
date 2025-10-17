import numpy as np
from scipy.stats import norm
"""
    A module containing functions to evaluate various greeks (with conventions) and retrieve parameters from delta
    Currently calculates Vanilla Greeks with formulas
    TODO - Extend to other options with finite difference methods
"""


class Greeks:

    def __init__(self, r, q, t, put=False) -> None:
        """
        Calculates Greeks and other related values for the BSM model

        Parameters
        ----------
        r : float
            The interest rate (FX: domestic rate)
        q : float
            The dividend rate (FX: foreign rate)
        t : float
            The term to maturity/tenor in day fractions of a year
        put : bool (DEFAULT = False)
            If true, option is a put, otherwise a call
        """
        self.r = r
        self.q = q
        self.t = t
        self.put = put

    def d1(self, S, vol, K=None) -> float:
        """
        Calculates d1, where N(d1) is how much more the present value of S exceeds the present value of K, given the option expires in the money

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)

        Returns
        -------
        float
            d1
        """
        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value d1")

        return (np.log(S / K) + self.t * (self.r - self.q +
                vol**2 / 2)) / (vol * np.sqrt(self.t))

    def d2(self, S, vol, K=None) -> float:
        """
        Calculates d2, where N(d2) is the probabilty that the option expires in the money

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float (DEFAULT = None)
            The strike price (FX: strike rate)

        Returns
        -------
        float
            d2
        """
        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value d2")

        return (np.log(S / K) + self.t * (self.r - self.q -
                vol**2 / 2)) / (vol * np.sqrt(self.t))

    def delta(self, S, vol, K=None, forward=False) -> float:
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

        Returns
        -------
        float
            The spot/forward delta of the option
        """
        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value delta")

        d1 = self.d1(S, vol, K)
        s = -1 if self.put else 1
        fw = self.q if forward else 0

        return s * norm.cdf(s * d1) * np.exp(-fw * self.t)

    def dual_delta(self, S, vol, K=None) -> float:
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

        Returns
        -------
        float
            The dual delta of the option
        """
        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value delta")

        d2 = self.d2(S, vol, K)
        s = -1 if self.put else 1

        return -s * norm.cdf(s * d2) * np.exp(-self.r * self.t)

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

        Returns
        -------
        float
            The premium-adjusted spot/forward put/call delta of the option
        """
        if K is None:
            K = S
        if vol is None:
            raise ValueError(
                "Please enter volatility to value premium-adjusted delta")

        d2 = self.d2(S, vol, K)
        s = -1 if self.put else 1
        fw = self.q if forward else 0

        return s * norm.cdf(s * d2) * K / S * np.exp((fw - self.r) * self.t)

    def vega(self, S, vol, K=None) -> float:
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

        Returns
        -------
        float
            The vega of the option
        """

        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value vega")

        d1 = self.d1(S, vol, K)

        return S * np.exp(-self.q * self.t) * np.sqrt(self.t) * norm.pdf(d1)

    def vanna(self, S, vol, K=None) -> float:
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

        Returns
        -------
        float
            The vanna of the option
        """

        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value vanna")

        d1 = self.d1(S, vol, K)
        d2 = self.d2(S, vol, K)

        return np.exp(self.q * self.t) * norm.pdf(d1) * d2 / vol

    def volga(self, S, vol, K=None) -> float:
        """
        Calculates the volga of the option,
        i.e the second derivative of the option price w.r.t volatility

        Parameters
        ----------
        S : float
            The spot price (FX: spot rate)
        vol : float
            The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
        K : float
            The strike price (FX: strike rate)
        forward : bool (DEFAULT = False)
            evaluate forward delta if true, else spot delta

        Returns
        -------
        float
            The volga of the option
        """

        if K is None:
            K = S
        if vol is None:
            raise ValueError("Please enter volatility to value volga")

        d1 = self.d1(S, vol, K)
        d2 = self.d2(S, vol, K)

        return S * np.exp(-self.q * self.t) * \
            np.sqrt(self.t) * norm.pdf(d1) * d1 * d2 / vol
