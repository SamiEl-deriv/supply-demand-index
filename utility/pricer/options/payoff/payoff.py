from abc import ABC, abstractmethod
from typing import Type
import numpy as np


class PayOff(ABC):
    """
    Base PayOff class
    """

    def __init__(self,
                 Strike: float,
                 **kwargs: float):
        """
        Parameters
        ----------
        Strike : float
            Strike price/rate
            Barrier price/rate if any, else None
        """
        self.__Strike = Strike
        self.__CriticalPoint = ()

    def set_critical_point(self, *kwargs):
        """
        Initialize a tuple of critical points:
        These are points where the payoff is discontinuous
        """
        self.__CriticalPoint = kwargs

    @property
    def critical_point(self) -> tuple:
        """
        Returns the tuple of critical points:
        These are points where the payoff is discontinuous

        Returns
        -------
        tuple[float]
            Tuple of critical points
        """

        return self.__CriticalPoint

    @property
    def strike(self) -> tuple:
        """
        Returns the strike

        Returns
        -------
        float
            Strike price
        """

        return self.__Strike

    @abstractmethod
    def __call__(self) -> float:
        """
        Abstract method for evaluating the payoff
        """
        raise NotImplementedError('Implementation required!')

    def onPath(self, paths: np.ndarray):
        """
        Payoff conditionned to the path of spot price followed
        by the underlying. By default this function return the
        payoff defined in __call__ method applied to the final
        spot values.

        Parameters
        ----------
        paths : np.ndarray[float]
            The 2D array where each lines correspond to a different
            path of the underlying spot value and each columns
            to a different time step. The last column is then
            the final spot value.

        Returns
        -------
        np.ndarray[float]
            the 1D array of the final payoff for each paths.
        """
        return self(paths[:, -1])


def DownAndOutDeco(cls: Type[object]) -> Type[object]:
    origin_onPath = cls.onPath

    def onPath(self, paths: np.ndarray):
        KO_mask = np.amin(paths, axis=1) > self.barrier
        payOffs = origin_onPath(self, paths) * KO_mask
        return payOffs

    cls.onPath = onPath
    return cls


def DownAndInDeco(cls: Type[object]) -> Type[object]:
    origin_onPath = cls.onPath

    def onPath(self, paths: np.ndarray):
        KO_mask = np.amin(paths, axis=1) < self.barrier
        payOffs = origin_onPath(self, paths) * KO_mask
        return payOffs

    cls.onPath = onPath
    return cls


def UpAndOutDeco(cls: Type[object]) -> Type[object]:
    origin_onPath = cls.onPath

    def onPath(self, paths: np.ndarray):
        KO_mask = np.amax(paths, axis=1) < self.barrier
        payOffs = origin_onPath(self, paths) * KO_mask
        return payOffs

    cls.onPath = onPath
    return cls


def UpAndInDeco(cls: Type[object]) -> Type[object]:
    origin_onPath = cls.onPath

    def onPath(self, paths: np.ndarray):
        KO_mask = np.amax(paths, axis=1) > self.barrier
        payOffs = origin_onPath(self, paths) * KO_mask
        return payOffs

    cls.onPath = onPath
    return cls


class PayOffDigitalPut(PayOff):
    """
    Digital Put payoff class

    Attributes
    ----------
    _Strike : float
        Strike price/rate
    """

    def __call__(self, Spot: float) -> float:
        """
        Returns the digital put payoff based on a spot price/rate

        DP(S) = 1 if S < K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Digital put payoff
        """
        return 1 * (Spot < self.strike)


class PayOffDigitalCall(PayOff):
    """
    Digital Call payoff class

    Attributes
    ----------
    _Strike : float
        Strike price/rate
    """

    def __call__(self, Spot: float) -> float:
        """
        Returns the digital call payoff based on a spot price/rate

        P(S) = 1 if S > K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Digital call payoff
        """
        return 1 * (Spot > self.strike)


class PayOffPut(PayOff):
    """
    Put payoff class

    Attributes
    ----------
    _Strike : float
        Strike price/rate
    """

    def __call__(self, Spot: float) -> float:
        """
        Returns the put payoff based on a spot price/rate

        P(S) = K-S if S < K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Vanilla put payoff
        """
        return np.maximum(self.strike - Spot, 0)


class PayOffCall(PayOff):
    """
    Call payoff class

    Attributes
    ----------
    _Strike : float
        Strike price/rate
    """

    def __call__(self, Spot: float) -> float:
        """
        Returns the call payoff based on a spot price/rate

        P(S) = S-K if S > K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Vanilla call payoff
        """
        return np.maximum(Spot - self.strike, 0)


@UpAndOutDeco
class PayOffUpOutCall(PayOff):
    """
    Up and Out Call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Up and Out Call payoff based on a spot price/rate

        P(S) = max(S-K,0)$Ind_{S<B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Up and Out call payoff
        """
        return np.maximum(Spot - self.strike, 0) * (Spot < self.__Barrier)


@UpAndInDeco
class PayOffUpInCall(PayOff):
    """
    Up and In Call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Up and In Call payoff based on a spot price/rate

        P(S) = max(S-K,0)$I_{S>B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Up and In call payoff
        """
        return np.maximum(Spot - self.strike, 0) * (Spot > self.__Barrier)


@DownAndOutDeco
class PayOffDownOutCall(PayOff):
    """
    Down and Out Call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Down and Out Call payoff based on a spot price/rate

        P(S) = max(S-K,0)*I_{S>B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Down and out call payoff
        """
        return np.maximum(Spot - self.strike, 0) * (Spot > self.__Barrier)


@DownAndInDeco
class PayOffDownInCall(PayOff):
    """
    Down and In Call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Down and In Call payoff based on a spot price/rate

        P(S) = max(S-K,0)$I_{S<B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Down and In call payoff
        """
        return np.maximum(Spot - self.strike, 0) * (Spot < self.__Barrier)


@DownAndOutDeco
class PayOffDownOutPut(PayOff):
    """
    Down and Out Put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Down and out put payoff based on a spot price/rate

        P(S) = max(K-S,0)*I_{S>B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Down and Out put payoff
        """
        return np.maximum(self.strike - Spot, 0) * (Spot > self.__Barrier)


@DownAndInDeco
class PayOffDownInPut(PayOff):
    """
    Down and In Put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Down and In Put payoff based on a spot price/rate

        P(S) = max(K-S,0)*I_{S<B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Down and In put payoff
        """
        return np.maximum(self.strike - Spot, 0) * (Spot < self.__Barrier)


@UpAndOutDeco
class PayOffUpOutPut(PayOff):
    """
    Up and Out Put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Up and Out Put payoff based on a spot price/rate

        P(S) = max(K-S,0)*I_{S<B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Up and Out put payoff
        """
        return np.maximum(self.strike - Spot, 0) * (Spot < self.__Barrier)


@UpAndInDeco
class PayOffUpInPut(PayOff):
    """
    Up and In Put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        self.__Barrier = Barrier

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the Up and In Put payoff based on a spot price/rate

        P(S) = max(K-S,0)*I_{S>B}

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Up and In put payoff
        """
        return np.maximum(self.strike - Spot, 0) * (Spot > self.__Barrier)


@UpAndOutDeco
class PayOffSharkfinCall(PayOff):
    """
    Sharkfin Call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        if Barrier > Strike:
            self.__Barrier = Barrier
        else:
            raise ValueError('Barrier should be greater than strike '
                             'for sharkfin call')

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the sharkfin call payoff based on a spot price/rate

        P(S) = S-K if S > K and S < B else 0

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Sharkfin call payoff
        """

        return np.maximum(Spot - self.strike, 0) * (Spot < self.__Barrier)


@DownAndOutDeco
class PayOffSharkfinPut(PayOff):
    """
    Sharkfin Put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-Out price/rate barrier
    """

    def __init__(self, Strike: float, Barrier: float):
        super().__init__(Strike, Barrier=Barrier)
        if Barrier < Strike:
            self.__Barrier = Barrier
        else:
            raise ValueError('Barrier should be lower than strike'
                             'for sharkfin put')

    @property
    def barrier(self):
        """
        Returns the KO barrier

        Returns
        -------
        float
            KO barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the sharkfin put payoff based on a spot price/rate

        P(S) = K-S if S < K and S > B else 0

        Parameters
        ----------
        strike : float
            Strike price/rate
        barrier : float
            Knock-Out price/rate barrier
        Returns
        -------
        float
            Sharkfin put payoff
        """
        return np.maximum(self.strike - Spot, 0) * (Spot > self.__Barrier)


@UpAndInDeco
class PayOffOneTouchCall(PayOff):
    """
    One touch call payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-in price/rate barrier
    """

    def __init__(self, Strike: float):
        super().__init__(Strike, Barrier=Strike)
        self.__Barrier = Strike

    @property
    def barrier(self):
        """
        Returns the KI barrier

        Returns
        -------
        float
            KI barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the digital call payoff based on a spot price/rate

        P(S) = 1 if S > K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Digital call payoff
        """
        return 1 * (Spot > self.strike)


@DownAndInDeco
class PayOffOneTouchPut(PayOff):
    """
    One touch put payoff class

    Attributes
    ----------
    strike : float
        Strike price/rate
    barrier : float
        Knock-in price/rate barrier
    """

    def __init__(self, Strike: float):
        super().__init__(Strike, Barrier=Strike)
        self.__Barrier = Strike

    @property
    def barrier(self):
        """
        Returns the KI barrier

        Returns
        -------
        float
            KI barrier
        """
        return self.__Barrier

    def __call__(self, Spot: float) -> float:
        """
        Returns the digital call payoff based on a spot price/rate

        P(S) = 1 if S > K else 0

        Parameters
        ----------
        Spot : float
            Spot price/rate

        Returns
        -------
        float
            Digital call payoff
        """
        return 1 * (Spot < self.strike)
