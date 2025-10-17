from deriv_quant_package.pricer.bsm import BSM
from deriv_quant_package.pricer.greeks import Greeks
from scipy.stats import norm
import numpy as np

"""
Helper functions for market conventions
"""


def ATM_delta(self, q, T, vol_ATM, forward=False, pa=False):
    """
    Retrieves the delta-neutral ATM delta
    Parameters
    ----------
    q : float
        The dividend rate (FX: foreign rate)
    T : float
        The term to maturity/tenor in day fractions of a year
    vol_ATM : float
        The ATM volatility
    put : bool
        The class values a put option if true and a call option if false
    forward : bool
        evaluate forward delta if true, else spot delta
    pa : bool
        get strike for premium-adjusted delta if true, else spot/forward delta

    Returns
    -------
    float
        The ATM delta
    """

    forward_factor = 0 if forward else -q
    pa_factor = -vol_ATM**2 / 2 if pa else 0
    s = -1 if put else 1

    return s / 2 * np.exp(self.T * (forward_factor + pa_factor))


def ATM_delta_neutral_strike(S, r, q, T, vol_ATM, pa=False):
    """
    Retrieves the delta neutral ATM strike
    Parameters
    ----------
    S : float
        The spot rate
    r : float
        The interest rate (FX: domestic rate)
    q : float
        The dividend rate (FX: foreign rate)
    T : float
        The term to maturity/tenor in day fractions of a year
    vol_ATM : float
        The ATM volatility
    pa : bool
        get strike for premium-adjusted delta if true, else spot/forward delta

    Returns
    -------
    float
        The ATM strike
    """

    pa_factor = -1 if pa else 1

    return S * np.exp((r - q) * T + pa_factor * vol_ATM**2 * T / 2)


def get_strike_from_delta(
        delta,
        r,
        q,
        vol,
        S,
        T,
        put=False,
        pa=False,
        forward=False,
        max_iter=5,
        tol=1e-8):
    """
    Retrieves a strike from delta (any of the 4 types)

    If non-premium adjusted, a formula is used.
    If premium adjusted, uses the Newton method to find an approximate strike (strikes may not be correct for sufficiently high delta)

    Parameters
    ----------
    delta : float
        The delta (spot/forward/premium adjusted)
    r : float
        The interest rate (FX: domestic rate)
    q : float
        The dividend rate (FX: foreign rate)
    T : float
        The term to maturity/tenor in day fractions of a year
    vol : float
        The volatility, if given numeric input, is flat, otherwise is retrieved from a volatility surface
    K : float
        The strike price (FX: strike rate)
    put : bool
        The class values a put option if true and a call option if false
    forward : bool
        evaluate forward delta if true, else spot delta

    Returns
    -------
    float
        A strike
    """

    s = -1 if put else 1
    factor = 1 if forward else np.exp(q * T)
    D = delta * factor
    diff = 0.00001

    # Calculate initial strike, is accurate if not premium adjusted
    K0 = S * np.exp((r - q) * T - s * norm.ppf(s * D)
                    * vol * np.sqrt(T) + vol**2 * T / 2)

    K = K0
    # Initialize BSM pricer
    BS_pricer = BSM(r=r, T=T, q=q, put=put)

    # Iterate to get accurate approximation of actual strike if premium
    # adjusted
    for i in range(max_iter * pa):
        # Get option price and delta according to current strike
        V1 = BS_pricer(S, vol, K)
        D1 = BS_pricer.greeks.delta(S, vol, K)

        # Get option price and delta to slightly shifted strike
        V2 = BS_pricer(S, vol, K * (1 + diff))
        D2 = BS_pricer.greeks.delta(S, vol, K * (1 + diff))

        # Calculate dDelta/dK for Newton-Raphson method
        dDelta_dK = (D2 - V2 / S - (D1 - V1 / S)) / (diff * K)
        K1 = K - (D1 - V1 / S - D) / dDelta_dK

        # Terminate if steps are close enough
        if abs(K - K1) < tol:
            K = K1
            break
        else:
            K = K1

    return K
