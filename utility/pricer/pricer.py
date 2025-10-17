import numpy as np
from typing import Union

# Import market data classes
from .market.market_variables import MarketSnapshot

from .market.interestrates.yieldcurvefactory import YieldCurveFactory
from .market.interestrates.refdict import yield_curve_types

from .market.volsurface.volsurface import VolSurface

# Import stochastic indices classes
from .market.stochastic_indices.stochastic_index_factory import StochasticIndexFactory

# Import Option classes and factory
from .options.OptionFactory import OptionFactory
from .options.payoff.PayOffFactory import PayOffFactory

# Import Boundary factory and class
from .pricing_engine.PDE.boundary.BoundaryFactory import BoundaryFactory

# Import PDE classes
from .pricing_engine.PDE.pde_params.pdebs import PDEBlackScholes

# Import FDM pricer classes
from .pricing_engine.PDE.scheme.FDM_scheme_factory import FDMschemeFactory

# Import paths generators factory, paths generators and MC setups
from .pricing_engine.MonteCarlo.MCEngine import MonteCarloEngine
from .pricing_engine.MonteCarlo.MCSetups import MCSetupsRefsDict
from .pricing_engine.MonteCarlo.returns_path_generators.RPGFactory import RPGRefDict

# Import Monte Carlo Engine
from .pricing_engine.MonteCarlo.MCEngine import MonteCarloEngine

# Import Option BSM Factory
import sys 
sys.path.append('../../RandD')
from RandD.Risk.OptionBSMFactory import *

"""
Dicts for user convenience
TODO: To somehow scrape classes from modules and assign appropriate strings
"""


class PDE_pricer:
    """
    Black-Scholes option pricer via the PDE (finite difference) pricing method.

    Attributes
    ----------
    pricer : FiniteDifference
        Main pricer object
    """
    def __init__(self, ref_spot : float, strike : float, riskless_rates : Union[np.ndarray, float], time_to_expiry : float, 
                 vols : Union[dict[int, dict[Union[float, int], float]], float], option : str, barrier : float = None, 
                 dividend_rates : Union[np.ndarray, float] = 0, 
                 american : bool = False, riskless_rate_type : str = "flat", dividend_rate_type : str = "flat", 
                 volsurface_type : str = "flat", volsurface_interp : str = None, market : str = "forex",
                 N_S : int = 100, N_t : int = 30,  scheme : str = "crank_nicolson", solver : str = "fast", boundary : str = "dirichlet",
                 grid_shift : bool = False,smoothing : float = 0.0,min_value_for_local_vol:float =0.0,gap_with_existing_points : float =0.1,
                 gap_between_new_points : float =0.1, number_new_points : int = 10):
        #TODO description for smoothing and min_value_for_local_vol
        """
        Parameters
        ----------
        ref_spot : float
            Reference spot
        strike : float
            Strike price/rate (K)
        riskless_rates : float
            Risk-free rate(s) (r)
            Refer to riskless_rate_type for format 
        time_to_expiry : float
            The (maximum) time-to-expiry/tenor (T)
        vols : float
            Volatilities (sigma)
            Refer to volsurface_interp for format 
        option : str
            The option type
            Currently supports ["call", "put", "digital_call", "digital_put","up_out_call","up_in_call",
            "down_out_call","down_in_call","down_out_put","down_in_put","up_out_put","up_in_put"
              "sharkfinKO_call", "sharkfinKO_put", "sharkfinXP_call", "sharkfinXP_put"]
        barrier : float
            Barrier price/rate (B)
        dividend_rates : float (DEFAULT : 0)
            The dividend rate(s) (q)
            Refer to dividend_rate_type for format 
        american : bool
            If true, value an American-style option.
            Otherwise value a European option
        riskless_rate_type : str (DEFAULT = "flat")
            Determines interpolation between different interest rate quotes. 
            * flat - Constant line
                Format : float | np.array([1, rate])
            * step - Piecewise constant curve
                Format : np.array([[T, rate] for T, rate in (T_list, rate_list)])
            * linear - Piecewise linear curve
                Format : np.array([[T, rate] for T, rate in (T_list, rate_list)])
        dividend_rate_type : str (DEFAULT = "flat")
            Determines interpolation between different dividend rate quotes. 
            * flat - Constant line
                Format : float | np.array([1, rate])
            * step - Piecewise constant curve
                Format : np.array([[T, rate] for T, rate in (T_list, rate_list)])
            * linear - Piecewise linear curve
                Format : np.array([[T, rate] for T, rate in (T_list, rate_list)])
        volsurface_type : str (DEFAULT = "flat")
            Determines interpolation between different dividend rate quotes. 
            * flat - Constant surface
                Format : float
            * quadratic - quadratic spline-interpolated surface
                Format : {1 : {x1 : mkt_vol1, .... xN : mkt_volN}, 365 : {x1 : mkt_vol1, .... xN : mkt_volN}, ...}
            * cubic - cubic spline-interpolated surface
                Format : {1 : {x1 : mkt_vol1, .... xN : mkt_volN}, 365 : {x1 : mkt_vol1, .... xN : mkt_volN}, ...}
            * vv - Vanna-Volga interpolated surface
                Format : {1 : {x1 : mkt_vol1, .... xN : mkt_volN}, 365 : {x1 : mkt_vol1, .... xN : mkt_volN}, ...}
        market : str (DEFAULT = "forex")
            The reference market. Currently placeholder value for future use
        N_x : int (DEFAULT = 100)
            The number of points in the spot-axis to use
        N_t : int (DEFAULT = 100)
            The number of points in the time-axis to use
        scheme : str (DEFAULT = "crank_nicolson")
            Specifies the scheme for the FDM to use.
            Currently supports ["explicit", "implicit", "crank-nicolson"]
        solver : str (DEFAULT = "fast")
            Specifies the solver to use in the FDM (Only applicable if scheme = "implicit" | "crank_nicolson")
            Currently supports ["fast", "gaussian"]
        boundary : str (DEFAULT = "dirichlet")
            Specifies the boundary condition style. Currently supports:
                dirichlet - value at boundary is pre-defined, i.e f(x,0) = x
                neumann - derivative at boundary is pre-defined i.e f'(x,0) = 1
        grid_shift : bool (DEFAULT= False)
            If true, apply grid shifting
        """

        # Parameter checking
        solvers = ["fast", "gaussian"]
        if solver not in solvers and scheme != "explicit":
            raise ValueError(f"Selected solver not supported\n Try one of {solvers} or leave blank for \"fast\"")

        self.volsurface_interp = volsurface_interp
        # Bar access from mixing of non-constant interest rates and VV until implemented
        if volsurface_interp == "vv" and not (isinstance(riskless_rates, float) or isinstance(dividend_rates, float)):
            raise NotImplementedError("Non-constant interest/dividend rates currently unsupported by Vanna-Volga interpolation")

        # Retrieve info from dicts
        riskless_rate_class = yield_curve_types[riskless_rate_type]
        dividend_rate_class = yield_curve_types[dividend_rate_type]

        # Register classes in factories
        YieldCurveFactory.RegisterYieldCurve(riskless_rate_type, riskless_rate_class)
        YieldCurveFactory.RegisterYieldCurve(dividend_rate_type, dividend_rate_class)

        # Create Instances
        payoff_to_use = PayOffFactory.CreatePayoff(option, Strike=strike, Barrier=barrier)
        FDMscheme = FDMschemeFactory.CreateFDM(scheme)

        # Construct market variables class
        riskless_rate_curve = YieldCurveFactory.CreateYieldCurve(riskless_rate_type, 
                                                                 market_data = riskless_rates if isinstance(riskless_rates, np.ndarray) 
                                                                 else np.array([[1, riskless_rates]]))
        dividend_rate_curve = YieldCurveFactory.CreateYieldCurve(dividend_rate_type, market_data = dividend_rates if isinstance(dividend_rates, np.ndarray) 
                                                                 else np.array([[1, dividend_rates]]))

        smile_kwargs = {}
        if volsurface_interp == "vv": 
            smile_kwargs["S"] = ref_spot

        if volsurface_interp == "TPS_local_vol":
            smile_kwargs["S"] = ref_spot  
            smile_kwargs["r"] = riskless_rates
            smile_kwargs["smoothing"]= smoothing
            smile_kwargs["N_t"]=N_t
            smile_kwargs["N_x"]=N_S
            smile_kwargs["min_value_for_local_vol"] = min_value_for_local_vol
            smile_kwargs["gap_with_existing_points"] = gap_with_existing_points
            smile_kwargs["gap_between_new_points"] = gap_between_new_points
            smile_kwargs["number_new_points"] = number_new_points

        volsurface = VolSurface(vols, volsurface_type, volsurface_interp,**smile_kwargs)
        market_snapshot = MarketSnapshot(market, riskless_rate_curve, dividend_rate_curve, volsurface)

        # Set critical point
        if grid_shift:
            critical_points = [strike]
            if "shark" in option: 
                critical_points.append(barrier)
            payoff_to_use.set_critical_point(*critical_points)

        # Initialize Option
        Option = OptionFactory.CreateOption(
            option,
            OptionPayoff   = payoff_to_use,
            Expiry         = time_to_expiry,
            MarketSnapshot = market_snapshot
        )

        # Initialize and calculate PDE solutions
        if volsurface_interp != "TPS_local_vol":
            self.pricer = PDEBlackScholes(
                Option       = Option,
                Current_Spot = ref_spot,
                N_x          = N_S,
                N_t          = N_t,
                American     = american,
                FDMclass     = FDMscheme,
                mode         = solver
            )
        else :
            self.pricer = generalizedPDEBlackScholes(
                Option       = Option,
                Current_Spot = ref_spot,
                N_x          = N_S,
                N_t          = N_t,
                American     = american,
                FDMclass     = FDMscheme,
                mode         = solver
            )

    def get_price(self, spot : float, tenor : float):
        """
        Returns the price of an option given an initial spot price/rate and a tenor

        Parameters
        ----------
        spot : float
            The spot price/rate
        tenor : float
            The tenor in years

        Returns
        -------
        float
            An option price
        """
        return self.pricer.get_price(spot, tenor)
    
    def get_price_curve(self, with_coords : bool = False):
        """
        Returns the price curve an option with respect to spot

        Parameters
        ----------
        with_coords : bool (DEFAULT = False)
            If true, return prices with corresponding S-ordinates

        Returns
        -------
        float
            Price curve at t=0
        """
        return self.pricer.get_solution_range(with_coords=with_coords)
    
    def get_price_surface(self, with_coords : bool = False):
        """
        Returns the entire grid of option prices over [S_min, S_max] x [0, T]

        where [S_min, S_max] is the spot range;
        [0, T] is the time range, up to the maximum tenor T

        Parameters
        ----------
        with_coords : bool (DEFAULT = False)
            If true, prices with corresponding (S,t)-coordinates

        Returns
        -------
        numpy.ndarray
            The option price surface in terms of 
                [[S, t, V(S,t)] for S in [S_min..S_max] and t in [0..T]] if with_coords = True
                [V(S,t) for S in [S_min..S_max] and t in [0..T]] otherwise
        """

        return self.pricer.get_surface(with_coords=with_coords)


class MC_pricer:
    """
    Pricer of options using Monte Carlo method.

    Attributes
    ----------
    pricer : MonteCarloEngine
        Main pricer object
    """

    def __init__(self,
                 ref_spot: float,
                 strike: float,
                 time_to_expiry: float,
                 option: str,
                 market: str,
                 numberOfSamples: int,
                 discount_rate: Union[np.ndarray, float] = 0.,
                 barrier: float = None,
                 path_generation: str = 'standard',
                 averaging_method: str = 'standard',
                 seed: int = None,
                 time_resolution: float = 1/(365*24*3600),
                 **params):
        """
        Parameters
        ----------
        ref_spot : float
            Reference spot
        strike : float
            Strike price/rate (K)
        riskless_rates : float
            Risk-free rate(s) (r)
            Refer to riskless_rate_type for format
        time_to_expiry : float
            The (maximum) time-to-expiry/tenor (T)
        option : str
            The option type
            Currently supports ["call", "put", "digital_call", "digital_put",
              "sharkfinKO_call", "sharkfinKO_put", "sharkfinXP_call", "sharkfinXP_put"]
        market : str
            The underlying market.
        numberOfSamples : int
            The number of simulation of the market evolution.
        barrier : float
            Barrier price/rate (B)
        path_generation : str
            The underlying market spot paths generation technique.
            Currently supports ["standard", "antithetic", "terminal stratification",
              "sobol", "halton"].
            Default is "standard".
        averaging_method : str
            The variance reduction technique used that acts on the pay off averaging
            after generating the paths. Such a method acts on the weight attributed
            to each simulation during the averaging for the option price computation.
            Currently supports ["standard", "CVoption", "CVunderlying"].
            Default is "standard".
        seed : int
            the seed of the random number generator used for the simulation.
        time_resolution : float
            The interval of times between two ticks.
        **params
            Additionnal parameters used for paths generation or averaging techniques
            NOTE: for now  only concerns the averaging techniques. Indeed CVoption
            needs to now the option payoff used as control variate and its price which
            are stored here as a key-value pair. 
        """

        if averaging_method not in MCSetupsRefsDict.keys():
            raise ValueError("Selected Monte Carlo averaging method not supported\n"
                             f"Try one of {list(MCSetupsRefsDict.keys())}")
        if (path_generation not in RPGRefDict.keys()):
            raise ValueError("Selected Monte Carlo returns paths generator not supported\n"
                             f"Try one of {list(MCSetupsRefsDict.keys())}")

        # Retrieve info from dicts
        mc_builder = MCSetupsRefsDict[averaging_method]

        # Initialize Payoff
        option_PO = PayOffFactory.CreatePayoff(option,
                                               Strike=strike,
                                               Barrier=barrier)

        # Initialize Market model
        market_model = StochasticIndexFactory.CreateStochIndex(market)

        # Initialize Paths generator
        RPG = RPGRefDict[path_generation]

        # Building and setting up the pricer
        self.__pricer = mc_builder(market_model,
                                   option_PO,
                                   time_to_expiry,
                                   ref_spot,
                                   numberOfSamples,
                                   discount_rate=discount_rate,
                                   dt=time_resolution,
                                   RPG=RPG,
                                   seed=seed,
                                   **params)

    @property
    def pricer(self) -> MonteCarloEngine:
        """
        Retrieve the pricer attribute.

        Returns
        -------
        MonteCarloEngine
            The pricer attribute.
        """
        return self.__pricer

    def get_price(self) -> float:
        """
        Get the estimated price of the option

        Returns
        -------
        float
            The estimated price of the option
        """
        return self.pricer.get_price()

    def get_simulated_paths(self) -> np.ndarray:
        """
        Retrieve the simulated paths of the market value

        Returns
        -------
        array
            the simulated paths of the market value
        """
        return self.pricer.paths

    def get_payOffs(self) -> np.ndarray:
        """
        Retrieve the pay offs of the option according to the simulated
        market paths.

        Returns
        -------
        array
            the pay offs of the option for each simulated paths.
        """
        return self.pricer.resulting_payOffs

    def redraw(self, **new_params):
        """
        Relaunch the simulation with new parameters.

        Parameters
        ----------
        **new_params:
            Key-value pairs of parameters of the new simulation. This
            parameters can be choosed within:
            {S_0: float, T: float, N: int, dt: float, seed: int}
        """
        self.pricer.redraw(**new_params)

    def reset(self):
        """
        Reset the simulation to the paths generated at the initiation of the 
        instance as well as its corresponding parameters.
        """
        self.pricer.reset()

class BSM_pricer:
    """
    Pricer of options using BSM formulas.

    """

    def __init__(self,
                 ref_spot: float,
                 strike: float,
                 time_to_expiry: float,
                 option: str,
                 market: str,
                 time_resolution: float = 1/(365*24*3600),
                 **params):
        """
        Parameters
        ----------
        ref_spot : float
            Reference spot
        strike : float
            Strike price/rate (K)
        time_to_expiry : float
            The (maximum) time-to-expiry/tenor (T)
        option : str
            The option type
            Currently supports ["call", "put", "digital_call", "digital_put",
              "sharkfinKO_call", "sharkfinKO_put", "sharkfinXP_call", "sharkfinXP_put"]
        market : str
            The underlying market.
        time_resolution : float
            The interval of times between two ticks.
        **params
            Additionnal parameters used for paths generation or averaging techniques
            NOTE: for now  only concerns the averaging techniques. Indeed CVoption
            needs to now the option payoff used as control variate and its price which
            are stored here as a key-value pair. 
        """
        self.spot = ref_spot
        self.time_to_expiry = time_to_expiry

        # Initialize Market model
        self.market_model = StochasticIndexFactory.CreateStochIndex(market)
        
        # Initialize pricer
        self.pricer = OptionBSMFactory.CreateOptionBSM(option, Strike = strike)
        
    def get_price(self) -> float:
        """
        Get the estimated price of the option

        Returns
        -------
        float
            The estimated price of the option
        """
        return self.pricer.price(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)
    
    def get_delta(self) -> float:
        """
        Get the estimated delta of the option

        Returns
        -------
        float
            The estimated delta of the option
        """
        return self.pricer.delta(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)

    def get_gamma(self) -> float:
        """
        Get the estimated gamma of the option

        Returns
        -------
        float
            The estimated gamma of the option
        """
        return self.pricer.gamma(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)

    def get_theta(self) -> float:
        """
        Get the estimated theta of the option

        Returns
        -------
        float
            The estimated theta of the option
        """
        return self.pricer.theta(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)

    def get_vega(self) -> float:
        """
        Get the estimated vega of the option

        Returns
        -------
        float
            The estimated vega of the option
        """
        return self.pricer.vega(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)

    def get_vanna(self) -> float:
        """
        Get the estimated vanna of the option

        Returns
        -------
        float
            The estimated vanna of the option
        """
        return self.pricer.vanna(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)

    def get_volga(self) -> float:
        """
        Get the estimated volga of the option

        Returns
        -------
        float
            The estimated volga of the option
        """
        return self.pricer.volga(Spot=self.spot, Time=self.time_to_expiry, Vol=self.market_model.volatility)