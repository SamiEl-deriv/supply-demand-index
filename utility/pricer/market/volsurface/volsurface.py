"""
Generates interpolated volatilities through time and spot
"""

from typing import Union, Type, Callable
from .volsmile import VolSmile
from .volsmile_flat import VolSmileFlat
from .volsmile_spline import VolSmileSpline
from .volsmile_vannavolga import VolSmileVannaVolga
from .volsmile_local_volatility_model import VolSmileLocalVol
import numpy as np

import warnings

VOLSURFACE_DICT : dict[str, VolSmile] = {
    "flat"      : VolSmileFlat,
    "quadratic" : VolSmileSpline,
    "cubic"     : VolSmileSpline,
    "vv"        : VolSmileVannaVolga,
    "TPS_local_vol" : VolSmileLocalVol
}

# TODO: Implement VolSurface Factory?
class VolSurface():

    def __init__(self, market_data : Union[dict[int, dict[Union[float, int], float]], float] , arg_type : str = "flat", interp_type : str = None, **smile_kwargs) -> None:

        """
        market_data : dict[int : dict[float | int : float]]
            The reference market quotes that will be used for interpolation. In particular -

            arg_type == strike/delta:
                * Format: {1 : {x1 : mkt_vol1, .... xN : mkt_volN}, 365 : {x1 : mkt_vol1, .... xN : mkt_volN}, ...}
                * strike will be interpolated directly here
                * delta will be converted to strike if vv is used
            arg_type == flat:
                * Format: float

        arg_type : str (DEFAULT = "flat")
            The argument to use when interpolating the volatility surface
            One of ['strike', 'delta', 'flat']
        
        interp_type : str (DEFAULT = None)
            Determines the interpolation type if arg_type is not flat.
            One of -
                * flat      : Constant across strikes/deltas
                * quadratic : Quadratic spline interpolation
                * cubic     : Cubic spline interpolation
                * vv        : Vanna-Volga implied volatility interpolation
                * TPS_local_vol : Thin plate spline interpolation + formula for local vol

        **smile_kwargs
            Additional arguments for the surface. Currently applicable only to Vanna-Volga interpolation and TPS interpolation
        """
        # Check valid arg type
        if arg_type not in (valid := ["strike", "delta", "flat"]):
            raise ValueError(f"Invalid surface type. Try one of {valid}")
        else: 
            self.arg_type = arg_type

        # Check valid volsurface types and market quotes for each arg type
        if arg_type == "flat":
            if interp_type is None:
                interp_type = "flat"
            if not isinstance(market_data, float):
                raise ValueError("market_data was not float with volsurface_type = flat")

        elif arg_type in ["strike", "delta"]:
            if interp_type not in (valid := ["quadratic", "cubic", "vv","TPS_local_vol"]):
                raise ValueError(f"Strike/Delta volatility interpolator selection invalid. Try one of {valid}")
            if len(market_data) == 0:
                raise ValueError(f"market_data is empty")
            
        else:
            raise ValueError(f"Volatility Surface argument type invalid. Try one of ['strike', 'delta', 'flat'] or leave blank for 'flat'")
        
        self.interp_type = interp_type
        self.market_data = market_data
        

        # Check smile_kwargs
        if interp_type == "vv":
            required = set(["S", "r", "q"])
        elif interp_type in ["quadratic", "cubic", "flat"]:
            required = set([])
        elif interp_type == "TPS_local_vol":
            required = set(["S",'r','smoothing'])

        if len(required - smile_kwargs.keys()) != 0:
            raise ValueError(f"Missing volsurface arguments for {interp_type}: {required}")
        elif interp_type == "TPS_local_vol":
            self.volsurface = VOLSURFACE_DICT[self.interp_type](self.market_data,**smile_kwargs)
        self.smile_kwargs = smile_kwargs
        # Spline requires cubic/quadratic argument
        if interp_type in ["quadratic", "cubic"]: self.smile_kwargs["type"] = interp_type
        # non-flat requires market data
        if interp_type in ["quadratic", "cubic", "vv"]: self.smile_kwargs["market_data"] = market_data

    def get_volatility(self, T : float, *args) -> float:
        """
        Retrieves interpolated volatilities from the volsurface

        Parameters
        ----------
        T : float
            The time to maturity of the contract
        *args
            A list of strikes/deltas to interpolate with
        
        Returns
        -------
        np.ndarray
            The requested volatilites for each strike
        """
        if self.interp_type != "TPS_local_vol":
            smile = self.get_smile(T)
        if self.interp_type in ["flat", "quadratic", "cubic"]:
            return smile(*args)
        elif self.interp_type == "vv":
            return smile(np.array(args))
        elif self.interp_type == "TPS_local_vol":
            return self.volsurface.get_vol(T,*args)

        else:
            # TODO: Implement delta & strike interpolators
            raise NotImplementedError("Strike & Delta interpolation not implemented yet")

    def get_smile(self, T : float) -> Callable[[float], float]:
        """
        Retrieves the time-interpolated volatility smile
        
        Parameters
        ----------
        T : float
            The time to maturity of the contract
        
        Returns
        -------
        Callable[[float], float]
            The inteprolated volatility smile at the requested T

        Notes/Todo
        ----------
        NOTE: This method does not support the vanna-volga method yet

        NOTE: Regarding spline interpolation, total variance interpolation is used here, but if intraday contracts are considered,
        we'll interpolate between (0,0) and (T,v^2T). TV interpolation implies that for any t in [0,T],
        v_t = v no matter what if we use the typical weights |T - T_j|/(T_{i+1} - T_i) where j = i,i+1.
        Looking at:
        Regentmarkets docs
        https://www.iasonltd.com/doc/old_rps/2007/2013_The_implied_volatility_surfaces.pdf page 8
        They do not seem to consider what happens in these cases

        TODO: Regarding spline interpolation, here the volsurface is assumed to have the same initial date as the contract
        which in general isn't possible as the volsurface in production is retrieved every 10 minutes. Furthermore, when
        forward contracts are considered, this assumption also doesn't hold.
        Modifications can be done so that we can take account of it:

        In general, we just set the result to TV_{from} - TV_{to} in this case
        
        TODO: Production code uses the calendar day weighting system as given in the same page of the pdf, which means that
        production vol does follow a sqrt trend (more or less) when intraday contracts are concerned.

        The current weights used are the usual linear distance between chosen point and edge points, which has the problem
        in the second note.

        Here, the discrete-ish calendar weights make this a bit more complicated. Seasonalized weights are even more so.
        Implementation requirements have to keep 4 cases in mind:

        Let the calendar/seasonalized weight corresponding to a time between an interval [T_i,T_{i+1}] be w_i(t)

        Case 1: From & To dates are on market_data points
        |<--------------------------------->|
        from/i                             to/i+1

            * TV_result = TV(T_{i+1}}) - TV(T_i)
        
        Case 2: From & To are between two consecutive market_data points
        |      <------------>               |
        i      from        to               i+1

            * TV_from/to = w_i(from/to) * (TV(T_{i+1}) - TV(T_i))
            * TV_result = TV_to - TV_from
                        = (w_i(from) - w_i(to)) * (TV(T_{i+1}) - TV(T_i))

        Case 3: From & To are in consecutive intervals:
        |      <-------------|------------->               |
        i      from          i+1          to               i+2

            * left  = TV_{i+1} - TV_from = (1 - w_i(from)) * TV(T_{i+1}) - w_i(from) * TV(T_i))
            * right = TV_to - TV_{i+1}   = w_{i+1}(to) * (TV(T_{i+2}) - TV{T_{i+1}))
            * TV_result = left + right
                        = w_{i+1}(to) * TV(T_{i+2}) + (1 - w_i(from) - w_{i+1}(to)) * TV_{T_{i+1}}
                          - w_i(from) * TV(T_i))

        Case 4: From & To are in non-consecutive intervals:
        |      <-------------|--------...--------|------------->               |
        i      from          i+1                 j            to               j+1

            * left   = (1 - w_i(from)) * TV(T_{i+1}) - w_i(from) * TV(T_i))
            * middle = TV(T_j) - TV(T_i)
            * right  = TV_to - TV_{j}   = w_j(to) * (TV(T_{j+1}) - TV{T_{j}))
            * TV_result = left + middle + right
                        = (1 - w_i(from)) * TV(T_{i+1}) - (1 + w_i(from)) + (1 - w_j(to)) * TV(T_j)
                          + w_{j}(to) * TV(T_{j+1})

        """
        # Initialize volsmile
        kwargs = self.smile_kwargs
        if self.interp_type == "vv": 
            kwargs["T"] = T
            S = kwargs.pop("S")
        volsmile_to_use = VOLSURFACE_DICT[self.interp_type](self.market_data,**kwargs)
        if self.interp_type == "flat": # No time interpolation required
            return volsmile_to_use.get_vol
        
        elif self.interp_type in ["quadratic", "cubic"]: # Time interpolation required here
            # Consider interpolation by 0
            print("interpolation")
            t_values = sorted(list(self.market_data.keys())).insert(0, 0)
            modified_market_data = self.market_data.copy()
            modified_market_data[0] = { key : 0 for key in modified_market_data[1].keys()}
            vol_dicts = [self.market_data for t in t_values]

            t_left = np.searchsorted(t_values, T) - 1

            # Get surrounding volsmiles

            # Get lambda function to return

        # volsmile_vannavolga is a child class of VV. To implement choosing T, requires refactoring of BSM, VV and vanna volga
        elif self.interp_type == "vv": 
            warnings.warn("VV smiles cannot handle multiple T at the moment, only at the beginning")
            return lambda K : volsmile_to_use.get_vol(K=K, S=S)
        

