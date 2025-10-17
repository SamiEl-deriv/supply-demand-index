from abc import ABC, abstractmethod
from ....options.options import *
import numpy as np

from typing import Type


class Boundary(ABC):
    """
    Abstract class specifying domain boundary conditions

    Attributes
    ----------
    _Option ; Option
        Contains relevant option information (payoff, r, q, etc.)
    S_max : float
        Maximum spot price
    S_min : float
        Minimum spot price
    """

    def __init__(self,
                 Option: Option,
                 Current_Spot: float):
        """
        Super initialization for all Boundary subclasses

        Parameters
        ----------
        Option : VanillaOption
            Option information
        Current_Spot : float
            Current spot price
        """
        self.__Option = Option
        # Initialize S-range
        half_xrange = 5 * \
            Option.volatility(Option.expiry, Option.strike)[
                0] * np.sqrt(Option.expiry)

        self.S_max = Current_Spot * np.exp(half_xrange)
        self.S_min = Current_Spot * np.exp(-half_xrange)

        # Throw error if K is out of bounds
        if Option.strike > self.S_max or Option.strike < self.S_min:
            raise ValueError(
                f"K is required to be within 5 standard deviations of S_current; {Current_Spot=}, {Option.strike=}")

    def set_spot_upper(self, S_max: float) -> None:
        """
        Set the upper spot boundary to the given value

        Parameters
        ----------
        S_max : float
            The new upper spot value
        """
        self.S_max = S_max

    def set_spot_lower(self, S_min: float) -> None:
        """
        Set the lower spot boundary to the given value

        Parameters
        ----------
        S_max : float
            The new lower spot value
        """
        self.S_min = S_min

    @abstractmethod
    def boundary_spot_lower(self):
        """
        Abstract method for setting up the lower spot boundary
        """
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def boundary_spot_upper(self):
        """
        Abstract method for setting up the upper spot boundary
        """
        raise NotImplementedError('Implementation required!')

    @property
    def option(self):
        """
        property: return the __Option attribute (object of type 'Option')
        """
        return self.__Option