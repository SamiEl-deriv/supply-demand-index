from .pde import PDEConvectionDiffusion
from ..fdm import FiniteDifference
from ....options.options import VanillaOption, SharkfinOption
from ..boundary.BoundaryFactory import BoundaryFactory

import warnings
import numpy as np
from typing import Type


class PDEBlackScholes(PDEConvectionDiffusion):
    """
    A class containing the various parameters for the Black-Scholes PDE with continuous dividends. Rewriting it in the necessary form:

    V_t = - (r-q) * S * V_S - vol^2 * S^2 * V_{SS} + r*V

    Attributes
    ----------
    _Option : VanillaOption
        The option information (K, r, q, etc.)
    _Smax : float
        The maximum S-value
    _Smin : float
        The minimum S-value
    _Boundary : Boundary
        The boundary information of the grid
    _FDM : Type[FiniteDifference]
        The finite difference machine which yields the solution of the PDE
    """

    def __init__(
            self,
            Option: VanillaOption,
            Current_Spot: float,
            N_x: int,
            N_t: int,
            American: bool,
            FDMclass: Type[FiniteDifference],
            mode="fast"):
        """
        Parameters
        ----------
        Option : VanillaOption
            The option information (K, r, q, etc.)
        Current_Spot : float
            The current spot (purely for reference purposes). Will return error if sufficiently far from strike
        N_x : int
            The number of space lattice points (assumed odd, otherwise will modify to the next odd number)
        N_t : int
            The number of time lattice points
        American : bool
            If positive, the option will be exercised immediately if the current payoff exceeds the current option value;
            otherwise treat as an European option
        FDMclass : Type[FiniteDifference]
            FDM Specification for pricing the option
        mode : str ("fast" | "gaussian")
            "fast": Tridiagonal matrix solver (Thomas algorithm) -- DEFAULT
            "gaussian": Gaussian matrix inversion (numpy.linalg.inv)

            For more details, check fdm.py
        """

        # Initialize Option information
        super().__init__(Option)

        if self.option.riskFreeRate(
                self.option.expiry) > self.option.volatility(
                self.option.expiry,
                self.option.strike):
            warnings.warn(
                "Volatility exceeds risk-free rate. May result in solution diverging")

        # Initialize boundary information
        # get the class of payoff in string format
        PayOffId = self.option.payOff.__class__.__name__
        self._Boundary = BoundaryFactory.CreateBoundary(
            PayOffId, Option=self.option, Current_Spot=Current_Spot)

        # Initialize S-range
        self._Smax = self._Boundary.S_max
        self._Smin = self._Boundary.S_min

        # Initialize pricer information
        self._FDM = FDMclass(self, N_x, N_t, mode, American)

    def coefficient_convection(self, x: float, t: float) -> float:
        """
        The convection coefficient (a = -(r-q)*S) of the BS PDE

        Parameters
        ----------
        x : float
            The current spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The convection coefficient for the given arguments
        """
        return -(self.option.riskFreeRate(self.option.expiry) -
                 self.option.dividendRate(self.option.expiry)) * x

    def coefficient_diffusion(self, x: float, t: float) -> float:
        """
        The diffusion coefficient (a = -0.5 * vol^2 * S^2) of the BS PDE

        Parameters
        ----------
        x : float
            The current spot price
        t : float
            The current time (up to self._Option._Expiry)

        Returns
        -------
        float
            The diffusion coefficient for the given arguments
        """
        return -0.5 * (self.option.volatility(self.option.expiry,
                       self.option.strike) * x)**2

    def coefficient_source(self, x: float, t: float) -> float:
        """
        The source term (Q = 0) of the BS PDE

        Parameters
        ----------
        x : float
            The current spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The source coefficient for the given arguments
        """
        return 0

    def coefficient_zero(self, x: float, t: float) -> float:
        """
        The zero coefficient (c = r) of the BS PDE

        Parameters
        ----------
        x : float
            The current spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The zero coefficient for the given argumentsw
        """
        return self.option.riskFreeRate(self.option.expiry)

    def boundary_spot_lower(self, t: float) -> float:
        """
        The lower spot boundary conditions (S=S_min < K)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        t : float
            The current time

        Returns
        -------
        float
            The price of the option at the lower spot boundary
        """
        return self._Boundary.boundary_spot_lower(t)

    def boundary_spot_upper(self, t: float) -> float:
        """
        The upper spot boundary conditions (S=S_max > K)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        t : float
            The current time

        Returns
        -------
        float
            The price of the option at the upper spot boundary
        """
        return self._Boundary.boundary_spot_upper(t)

    def init_cond(self, x: float) -> float:
        """
        The terminal conditions, i.e the payoff of the option

        Parameters
        ----------
        x : float
            The current spot

        Returns
        -------
        float
            The price of the option a expiry
        """
        return self.option.payOff(x)

    def get_S_range(self) -> np.ndarray:
        """
        Retrieves the space-lattice points

        Returns
        -------
        numpy.ndarray
            The x-ordinates of the grid
        """
        return self._FDM.get_x_range()

    def get_t_range(self) -> np.ndarray:
        """
        Retrieves the time-lattice points

        Returns
        -------
        numpy.ndarray
            The t-ordinates of the grid
        """
        return self._FDM.get_t_range()

    def get_surface(self, with_coords=False) -> np.ndarray:
        """
        Retrieves a populated grid - solution to the pde

        Returns
        -------
        numpy.ndarray
            The solution grid if it has been populated
        """
        return self._FDM.get_surface(with_coords=with_coords)

    def get_solution_range(self, with_coords=False) -> np.ndarray:
        """
        Retrieves the solution to the PDE at t=0

        Parameters
        ----------
        with_coords : bool (DEFAULT = False)
            If true, return numpy array containing S-ordinates and solution -- elements of form [S, f(S,0)]
            Else return numpy array of f(S,0)

        Returns
        -------
        numpy.ndarray
            The solution at time t=0 with or without coordinates
        """
        return self._FDM.get_solution_range(with_coords=with_coords)

    def get_price(self, Spot: float, Tenor: float) -> float:
        """
        Returns the price at the given spot and time

        Parameters
        ----------
        S : float
            The starting spot price/rate
        t : float
            The time to expiry of the contract

        Returns
        -------
        float
            The interpolated option price
        """

        # Cast given tenor to t = Max_Tenor - Tenor
        max_tenor = self.option.expiry

        if Tenor <= max_tenor:
            t = max_tenor - Tenor
        else:
            raise ValueError(
                f"Tenor is out of range: min_tenor = 0, max_tenor = {max_tenor}")

        return self._FDM.get_value(Spot, t)
