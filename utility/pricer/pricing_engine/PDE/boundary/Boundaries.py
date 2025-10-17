from .Boundary import Boundary
from ....options.options import *
import numpy as np

from typing import Type


class BoundaryCall(Boundary):
    """
    Boundary specifications for European (linear) call options
    """

    def boundary_spot_lower(self, t):
        """
        Returns lower spot vanilla call boundary value for a given t

        C(S_min,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range,
        we can safely assume 0 for the lower boundary conditions
        """
        return 0  # self._Option.OptionPayOff(self.S_min)* np.exp(self._Option._RiskFreeRate(self._Option._Expiry) * (self._Option._Expiry - t))

    def boundary_spot_upper(self, t):
        """
        Returns upper spot vanilla call boundary value for a given t

        C(S_max,t) = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}

        Here, consider the limiting behaviour of the BS equation as S -> infinity
        as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        C_upper(t) = lim{S -> infty} (S * e^{-q*(T-t)} * N(d_1) - K * e^{-r*(T-t)} * N(d_2))
                   = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_max, t) = 1

        Or a null-gamma condition:
        Gamma(S_max, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix
        """

        #  One alternative is payoff * foreign discount rate, but this doesn't exactly reflect long term behaviour, especially if T is large:
        # return self._PayOff(self.S_max)*
        # np.exp(-self._Option._RiskFreeRate(self._Option._Expiry) *
        # (self._Option._Expiry - t))

        spot_with_dividend = self.S_max * \
            np.exp(- self.option.dividendRate(self.option.expiry)
            * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return spot_with_dividend - discounted_strike


class BoundaryPut(Boundary):
    """
    Boundary specifications for European (linear) put options
    """

    def boundary_spot_lower(self, t):
        """
        Returns lower spot vanilla put boundary value for a given t

        P(S_min,t) = K * e^{-r*(T-t)} - S * e^{-q*(T-t)}

        Here, consider the limiting behaviour of the BS equation as S -> 0 as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        P_lower(t) = lim{S -> 0} (K * e^{-r*(T-t)} * N(-d_2) - S * e^{-q*(T-t)} * N(-d_1))
                   = K * e^{-r*(T-t)}

        However, considering that S_min is unlikely to be zero,

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_min, t) = 1

        Or a null-gamma condition:
        Gamma(S_min, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix

        refs (implementation):
        https://math.stackexchange.com/questions/2706701/neumann-boundary-conditions-in-finite-difference
        https://math.stackexchange.com/questions/2346861/using-finite-difference-method-for-1d-diffusion-equation

        refs (ghost points):
        https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/masterdegraaf.pdf
        http://www.math.toronto.edu/mpugh/Teaching/Mat1062/notes2.pdf
        """
        spot_with_dividend = self.S_min * \
            np.exp(-self.option.dividendRate(self.option.expiry)
                   * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return discounted_strike - spot_with_dividend

    def boundary_spot_upper(self, t):
        """
        Returns upper spot vanilla put boundary value for a given t

        P(S_max,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range, so we can safely assume 0 for the upper boundary conditions
        """
        return 0


class BoundaryUpOutCall(Boundary):
    """
    Boundary specifications for Up and Out call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_max = np.minimum(Option.barrier, self.S_max)
        self.S_min = 0

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        C(0,t) = 0
        see :  https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf

        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Up and Out call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        C_(S_max,t) = 0

        see : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf

        """

        return 0


class BoundaryUpInCall(Boundary):
    """
    Boundary specifications for Up and In call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_max = np.minimum(Option.barrier, self.S_max)
        self.S_min = 0

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        C(0,t) = 0
        see : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf

        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Up and In call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        C_(S_max,t) = 0 if B <= K
                     = BlachScholes(B,K,r,q,T-t,sigme) else

        see : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf
        """
        if self.option.barrier <= self.option.strike:
            return 0
        else:
            Put = False
            market_quote = None
            black_scholes_formula = BSM(
                self.option.riskFreeRate(0),
                self.option.expiry - t,
                self.option.dividendRate(0),
                market_quote,
                Put)
            return black_scholes_formula.get_price(
                self.option.barrier, self.option.strike, self.option.volatility(
                    self.option.expiry - t, self.option.strike))


class BoundaryDownOutCall(Boundary):
    """
    Boundary specifications for Down and Out call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_min = np.maximum(Option.barrier, self.S_min)

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        C(B,t) = 0
        see : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf

        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Down and Out call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        C(t,S) ~ S when S is big
        see : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf
        """

        return self.S_max


class BoundaryDownInCall(Boundary):
    """
    Boundary specifications for Down and In call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_min = np.maximum(Option.barrier, self.S_min)

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        C(B,t) = Bl(B,K,r,q,T-t,sigma)
        See : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf

        """
        Put = False
        market_quote = None
        black_scholes_formula = BSM(
            self.option.riskFreeRate(0),
            self.option.expiry - t,
            self.option.dividendRate(0),
            market_quote,
            Put)
        return black_scholes_formula.get_price(
            self.option.barrier, self.option.strike, self.option.volatility(
                self.option.expiry - t, self.option.strike))

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Down and In call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        C(t,S)~S when S is big
        See : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf
        """

        return self.S_max


class BoundaryDownOutPut(Boundary):
    """
    Boundary specifications for Down and Out put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_min = np.maximum(Option.barrier, self.S_min)

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        P(B,t) = 0
        See : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf

        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Down and out put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        P(t,S)-->0 when S is big
        See : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf
        """

        return 0


class BoundaryDownInPut(Boundary):
    """
    Boundary specifications for Down and in put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_min = np.maximum(Option.barrier, self.S_min)

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        P(B,t) = Bl_put(B,K,r,q,T-t,sigma) if B< K 0 else
        See : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf

        """
        if self.option.barrier < self.option.strike:
            Put = True
            market_quote = None
            black_scholes_formula = BSM(
                self.option.riskFreeRate(0),
                self.option.expiry - t,
                self.option.dividendRate(0),
                market_quote,
                Put)
            return black_scholes_formula.get_price(
                self.option.barrier, self.option.strike, self.option.volatility(
                    self.option.expiry - t, self.option.strike))
        else:
            return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Down and In put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        P(t,S)-->0 when S is big
        See : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf
        """

        return 0


class BoundaryUpOutPut(Boundary):
    """
    Boundary specifications for Up and out put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_max = np.minimum(Option.barrier, self.S_max)
        self.S_min = 0

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        P(0,t) =e^(-r(T-t))K
        See : https://empslocal.ex.ac.uk/people/staff/NPByott/teaching/FinMaths/2005/black-scholes.pdf
        """

        return np.exp(-self.option.riskFreeRate(t) *
                      (self.option.expiry - t)) * self.option.strike

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Up and Out put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        P(t,B) = 0
        See : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf
        """

        return 0


class BoundaryUpInPut(Boundary):
    """
    Boundary specifications for Up and in put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: BarrierOption, Current_Spot: float):
        super().__init__(Option, Current_Spot)
        self.S_max = np.minimum(Option.barrier, self.S_max)
        self.S_min = 0

    def boundary_spot_lower(self, t):
        """
        Returns lower spot  boundary value for a given t:

        P(0,t) = 0
        See : This one I did not find a reference. But you can compute it
        by hand taking the limit as S-->0 of the valuation formula for a
        up in put.
        """

        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot boundary value for a given t for Up and In put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached

        P(t,B) = Bl_put(B,K,r,T-t,sigma)
        See : https://personal.ntu.edu.sg/nprivault/MA5182/barrier-options.pdf
        """
        Put = True
        market_quote = None
        black_scholes_formula = BSM(
            self.option.riskFreeRate(0),
            self.option.expiry - t,
            self.option.dividendRate(0),
            market_quote,
            Put)
        return black_scholes_formula.get_price(
            self.option.barrier, self.option.strike, self.option.volatility(
                self.option.expiry - t, self.option.strike))


class BoundaryDigitalCall(Boundary):
    """
    Boundary specifications for digital call options
    """

    def boundary_spot_lower(self, t):
        """
        Returns lower spot digital call boundary value for a given t

        DC(S_min,t) = 0

        V - We expect the strike to be within the given range, so we can safely assume 0 for the lower boundary conditions
        """
        return 0  # self._Option.OptionPayOff(self.S_min)* np.exp(self._Option._RiskFreeRate(self._Option._Expiry) * (self._Option._Expiry - t))

    def boundary_spot_upper(self, t):
        """
        Returns upper spot digital call boundary value for a given t

        DC(S_max,t) = e^{-r*(T-t)}

        Here, consider the limiting behaviour of the BS formula for the Digital Call when S -> infinity as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        C_upper(t) = lim{S -> infty} e^{-r*(T-t)} * N(d_2)
                   = e^{-r*(T-t)}

        TODO : To be discussed / thoughts for the future
        V - These conditions are up to debate

        refs (implementation):
        https://math.stackexchange.com/questions/2706701/neumann-boundary-conditions-in-finite-difference
        https://math.stackexchange.com/questions/2346861/using-finite-difference-method-for-1d-diffusion-equation

        refs (ghost points):
        https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/masterdegraaf.pdf
        http://www.math.toronto.edu/mpugh/Teaching/Mat1062/notes2.pdf
        """
        return np.exp(-self.option.riskFreeRate(self.option.expiry)
                      * (self.option.expiry - t))


class BoundaryDigitalPut(Boundary):
    """
    Boundary specifications for digital put options
    """

    def boundary_spot_lower(self, t):
        """
        Returns lower spot digital put boundary value for a given t

        DP(S_min,t) = e^{-r*(T-t)}

        Here, consider the limiting behaviour of the BS formula for the Digital Call when S -> 0 as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        DP+lower(t) = lim{S -> 0} e^{-r*(T-t)} * N(-d_2)
                   = e^{-r*(T-t)}

        TODO : To be discussed / thoughts for the future
        V - These conditions are up to debate
        """
        return np.exp(-self.option.riskFreeRate(self.option.expiry)
                      * (self.option.expiry - t))

    def boundary_spot_upper(self, t):
        """
        Returns upper spot digital put boundary value for a given t

        DP(S_max,t) = 0

        TODO : To be discussed / thoughts for the future
        V - These conditions are up to debate
        """
        return 0


class BoundarySharkfinKOCall(Boundary):
    """
    Boundary specifications for Sharkfin call options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: SharkfinOption, Current_Spot: float):
        """
        Redefine S_max if it is above the KO barrier
        Redefine S_min to 0.
        TODO: find a better way of redefining these boundaries
        """
        super().__init__(Option, Current_Spot)
        self.S_max = np.minimum(Option.barrier, self.S_max)

    def boundary_spot_lower(self, t):
        """
        Returns lower spot sharkfin call boundary value for a given t

        C_sf(S_min,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range, so we can safely assume 0 for the lower boundary conditions
        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot sharkfin call boundary value for a given t:

        C_sf(S_max,t) = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}  if S_max < Barrier
                        0                                    if S_max > Barrier

        Here, consider the limiting behaviour of the BS equation as S -> infinity as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        C_upper(t) = lim{S -> infty} (S * e^{-q*(T-t)} * N(d_1) - K * e^{-r*(T-t)} * N(d_2))
                   = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_max, t) = 1

        Or a null-gamma condition:
        Gamma(S_max, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix
        """

        #  One alternative is payoff * foreign discount rate, but this doesn't exactly reflect long term behaviour, especially if T is large:
        # return self._PayOff(self.S_max)*
        # np.exp(-self._Option._RiskFreeRate(self._Option._Expiry) *
        # (self._Option._Expiry - t))

        spot_with_dividend = self.S_max * \
            np.exp(-self.option.dividendRate(self.option.expiry)
                   * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return (spot_with_dividend - discounted_strike) * \
            (self.S_max < self.option.barrier)


class BoundarySharkfinKOPut(Boundary):
    """
    Boundary specifications for sharkfin put options with immediate Knock-Out feature: contract value vanishes and expire as soon as the barrier is breached
    """

    def __init__(self, Option: SharkfinOption, Current_Spot: float):
        """
        Redefine S_min if it is below the KO barrier
        TODO: find a better way of redefining these boundaries
        """
        super().__init__(Option, Current_Spot)
        self.S_min = np.maximum(Option.barrier, self.S_min)

    def boundary_spot_lower(self, t):
        """
        Returns upper spot sharkfin call boundary value for a given t:

        P_sf(S_max,t) = K * e^{-r*(T-t)} - S * e^{-q*(T-t)}  if S_min > Barrier
                        0                                    if S_min < Barrier

        Here, consider the limiting behaviour of the BS equation as S -> 0 as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        P_lower(t) = lim{S -> 0} (K * e^{-r*(T-t)} * N(-d_2) - S * e^{-q*(T-t)} * N(-d_1))
                   = K * e^{-r*(T-t)}

        However, considering that S_min is unlikely to be zero,

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_min, t) = 1

        Or a null-gamma condition:
        Gamma(S_min, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix
        """

        #  One alternative is payoff * foreign discount rate, but this doesn't exactly reflect long term behaviour, especially if T is large:
        # return self._PayOff(self.S_max)*
        # np.exp(-self._Option._RiskFreeRate(self._Option._Expiry) *
        # (self._Option._Expiry - t))

        spot_with_dividend = self.S_min * \
            np.exp(-self.option.dividendRate(self.option.expiry)
                   * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return (discounted_strike - spot_with_dividend) * \
            (self.S_min > self.option.barrier)

    def boundary_spot_upper(self, t):
        """
        Returns upper spot sharkfin put boundary value for a given t

        P_sf(S_min,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range, so we can safely assume 0 for the lower boundary conditions
        """
        return 0


class BoundarySharkfinXPCall(Boundary):
    """
    Boundary specifications for Sharkfin call options with Knock-Out happening only at expiry: at the expiry if the spot price is beyond the barrier, the payoff is zero.
    """

    def __init__(self, Option: SharkfinOption, Current_Spot: float):
        """
        Redefine S_max to be far above (2 std) than the barrier
        TODO: find a better way of redefining these boundaries
        """
        super().__init__(Option, Current_Spot)
        self.S_max = Option.barrier * \
            np.exp(2 * Option.volatility(Option.expiry,
                   Option.strike) * np.sqrt(Option.expiry))

    def boundary_spot_lower(self, t):
        """
        Returns lower spot sharkfin call boundary value for a given t

        C_sf(S_min,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range, so we can safely assume 0 for the lower boundary conditions
        """
        return 0

    def boundary_spot_upper(self, t):
        """
        Returns upper spot sharkfin call boundary value for a given t:

        C_sf(S_max,t) = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}  if S_max < Barrier
                        0                                    if S_max > Barrier

        Here, consider the limiting behaviour of the BS equation as S -> infinity as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        C_upper(t) = lim{S -> infty} (S * e^{-q*(T-t)} * N(d_1) - K * e^{-r*(T-t)} * N(d_2))
                   = S * e^{-q*(T-t)} - K * e^{-r*(T-t)}

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_max, t) = 1

        Or a null-gamma condition:
        Gamma(S_max, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix
        """

        #  One alternative is payoff * foreign discount rate, but this doesn't exactly reflect long term behaviour, especially if T is large:
        # return self._PayOff(self.S_max)*
        # np.exp(-self._Option._RiskFreeRate(self._Option._Expiry) *
        # (self._Option._Expiry - t))

        spot_with_dividend = self.S_max * \
            np.exp(-self.option.dividendRate(self.option.expiry)
                   * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return (spot_with_dividend - discounted_strike) * \
            (self.S_max < self.option.barrier)


class BoundarySharkfinXPPut(Boundary):
    """
    Boundary specifications for sharkfin put options with Knock-Out happening only at expiry: at the expiry if the spot price is below the barrier, the payoff is zero.
    """

    def __init__(self, Option: SharkfinOption, Current_Spot: float):
        """
        Redefine S_min to be far lower (2 std) than the barrier
        TODO: find a better way of redefining these boundaries
        """
        super().__init__(Option, Current_Spot)
        self.S_min = Option.barrier * \
            np.exp(-2 * Option.volatility(Option.expiry,
                   Option.strike) * np.sqrt(Option.expiry))

    def boundary_spot_upper(self, t):
        """
        Returns upper spot sharkfin call boundary value for a given t

        P_sf(S_min,t) = 0

        TODO : To be discussed
        V - We expect the strike to be within the given range, so we can safely assume 0 for the lower boundary conditions
        """
        return 0

    def boundary_spot_lower(self, t):
        """
        Returns upper spot sharkfin call boundary value for a given t:

        P_sf(S_max,t) = K * e^{-r*(T-t)} - S * e^{-q*(T-t)}  if S_min > Barrier
                        0                                    if S_min < Barrier

        Here, consider the limiting behaviour of the BS equation as S -> 0 as we expect the limiting behaviour of the solution to be the same
        regardless of the method used to obtained it (Uniqueness of PDE solutions).

        P_lower(t) = lim{S -> 0} (K * e^{-r*(T-t)} * N(-d_2) - S * e^{-q*(T-t)} * N(-d_1))
                   = K * e^{-r*(T-t)}

        However, considering that S_min is unlikely to be zero,

        TODO : To be discussed / thoughts for the future
        V - Technically, the above Dirichlet Conditions can introduce errors if S is not sufficiently large (which is less likely due to the fact that it is 5 stds from the current spot) or if
        T is sufficiently large.
        That being the case, we may want to consider replacing it with Neumann conditions as it doesn't care what the value of the option price is.

        We could consider either a delta condition:

        Delta(S_min, t) = 1

        Or a null-gamma condition:
        Gamma(S_min, t) = 0

        Note that both these conditions have to be incorporated within each step_march in fdm.py as it changes the underlying step matrix
        """

        #  One alternative is payoff * foreign discount rate, but this doesn't exactly reflect long term behaviour, especially if T is large:
        # return self._PayOff(self.S_max)*
        # np.exp(-self._Option._RiskFreeRate(self._Option._Expiry) *
        # (self._Option._Expiry - t))

        spot_with_dividend = self.S_min * \
            np.exp(-self.option.dividendRate(self.option.expiry)
                   * (self.option.expiry - t))
        discounted_strike = self.option.strike * \
            np.exp(-self.option.riskFreeRate(self.option.expiry)
                   * (self.option.expiry - t))
        return (discounted_strike - spot_with_dividend) * \
            (self.S_min > self.option.barrier)
