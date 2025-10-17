from .payoff.payoff import PayOff
from ..market.market_variables import MarketSnapshot


class Option:
    """
    Option class

    Attributes
    ----------
    _OptionPayOff : PayOff
        The option's payoff structure
    _Expiry : float
        The time to expiry
    _RiskFreeRate : float
        The risk-free rate
    _DividendRate : float
        The dividend rate
    _Volatility : float
        The volatility
    """

    def __init__(self,
                 OptionPayoff: PayOff,
                 Expiry: float,
                 MarketSnapshot: MarketSnapshot):
        """
        Parameters
        ----------
        OptionPayoff : Payoff
            A payoff instance
        Expiry : float
            The time to expiry
        RiskFreeRate : float
            The risk-free rate
        DividendRate : float
            The dividend rate
        Volatility : float
            The volatility
        """

        self.__OptionPayOff = OptionPayoff
        self.__Expiry = Expiry
        self.__RiskFreeRate = MarketSnapshot.get_interest_rate
        self.__DividendRate = MarketSnapshot.get_dividend_rate
        self.__Volatility = MarketSnapshot.get_volatility

    @property
    def payOff(self) -> PayOff:
        """
        Returns the payoff object

        Returns
        -------
        PayOff
            The option's payoff object
        """
        return self.__OptionPayOff

    @property
    def expiry(self) -> float:
        """
        Returns the time to expiry

        Returns
        -------
        float
            The time to expiry
        """
        return self.__Expiry

    @property
    def strike(self) -> float:
        """
        Returns the strike of the option (payoff)

        Returns
        -------
        float
            The strike price/rate
        """
        return self.payOff.strike

    @property
    def volatility(self):
        """
        Returns the volatility of the underlying

        Returns
        -------
        float
            The volatility
        """
        return self.__Volatility

    @property
    def riskFreeRate(self):
        """
        Returns the risk free interest rate of the underlying

        Returns
        -------
        float
            The volatility
        """
        return self.__RiskFreeRate

    @property
    def dividendRate(self):
        """
        Returns the dividend rate of the underlying

        Returns
        -------
        float
            The volatility
        """
        return self.__DividendRate


class VanillaOption(Option):
    pass


class DigitalOption(Option):
    pass


class BarrierOption(Option):

    @property
    def barrier(self) -> float:
        """
        Returns the barrier of the option (payoff)

        Returns
        -------
        float
            The KO/KI price/rate barrier
        """
        return self.payOff.barrier


class SharkfinOption(Option):

    @property
    def barrier(self) -> float:
        """
        Returns the barrier of the option (payoff)

        Returns
        -------
        float
            The KO price/rate barrier
        """
        return self.payOff.barrier
