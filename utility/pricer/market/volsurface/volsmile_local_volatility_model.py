import numpy as np
from typing import Union, Type, Callable
from scipy.interpolate import Rbf,RegularGridInterpolator
from .volsmile import VolSmile
from .link_local_implied_vol import local_vol_as_function_of_implied_vol
import matplotlib.pyplot as plt

def extend(volsurface,gap_with_existing_points, gap_between_new_points,number_new_points):
    """Here, volsurface is a n*3 array representing a list of 
    points ex (Ti, Kj, sigma(Ti,Kj))"""
    tenors = np.unique(volsurface[:,0])
    list_mins = []
    list_maxs =[]
    for tenor in tenors:
        indexes = np.where(volsurface[:,0]==tenor)
        vol_smile = volsurface[indexes[0],:]
        min_spot = np.min(vol_smile[:,1])
        index_min_spot = np.where(vol_smile[:,1] == min_spot)
        vol_min_spot = vol_smile[index_min_spot,2]
        max_spot = np.max(vol_smile[:,1])
        index_max_spot = np.where(vol_smile[:,1] == max_spot)
        vol_max_spot = vol_smile[index_max_spot,2]
        for i in range(number_new_points):
            list_mins.append([tenor,min_spot-gap_with_existing_points-i*gap_between_new_points,vol_min_spot[0][0]])
            list_maxs.append([tenor,max_spot+gap_with_existing_points+i*gap_between_new_points,vol_max_spot[0][0]])
    new_points = np.array(list_mins+list_maxs)
    return np.concatenate((volsurface,new_points))

class VolSmileLocalVol(VolSmile):
    """
    A class that interpolates/extrapolates volatility using thin plate interpolation and the formula that links
    local volatility to implied volatility
    """

    def __init__(self, market_data : Union[dict[int, dict[Union[float, int], float]], float],**kwargs) -> None:
        """
        Constructs the thin plate spline used for interpolation
        Thin plate spline requires at least 3 points

        Parameters
        ----------
        smoothing : float
            A float containing the smoothing parameter for the interpolation
        market_data : dict(float : float)
            A dictionary containing strike/delta/moneyness-volatility pairs.
            * Format: {1 : {x1 : mkt_vol1, .... xN : mkt_volN}, 365 : {x1 : mkt_vol1, .... xN : mkt_volN}, ...}
        """
        self.market_data = market_data
        self.type = type
        self.initial_spot = kwargs["S"]
        self.interest_rate = kwargs["r"]
        self.smoothing = kwargs["smoothing"]
        self.N_x = kwargs["N_x"]
        self.N_t = kwargs["N_t"]
        self.min_value_for_local_vol = kwargs["min_value_for_local_vol"]
        self.smoothing = kwargs["smoothing"]
        self.gap_with_existing_points = kwargs["gap_with_existing_points"]
        self.gap_between_new_points = kwargs["gap_between_new_points"]
        self.number_new_points = kwargs["number_new_points"]

        # Catch exceptions
        if self.initial_spot <0:
            raise ValueError("Initial spot must be >=0")
        if self.smoothing < 0 :
            raise ValueError("Smoothing parameter must be >=0")
        if self.min_value_for_local_vol < 0 :
            raise ValueError("min_value_for_local_vol must be >=0")
        if not isinstance(self.N_t,int):
            raise ValueError("N_t must be an integer")
        if not isinstance(self.N_x,int):
            raise ValueError("N_x must be an integer")
        if self.gap_with_existing_points<0:
            raise ValueError("Gap with existing points must be >=0")
        if self.gap_between_new_points <0:
            raise ValueError("Gap between new points must be >=0")
        if not isinstance(self.number_new_points,int):
            raise ValueError("number of poitns must be an integer")
        
        #get the values of market data in an array
        #collecting the values for the edges of the volsurface
        tenor_list = []
        moneyness_list = []
        impvol_list = []
        for x in market_data.keys():
            for y, z in market_data[x].items():
                tenor_list.append(x)
                moneyness_list.append(y)
                impvol_list.append(z)
       
        #construct the arrays for the interpolation
        tenor,moneyness,impvol = np.array(tenor_list),np.array(moneyness_list),np.array(impvol_list)
        impvol_points = np.column_stack((tenor,moneyness,impvol))
        
        # extend the volatility surface
        impvol_points = extend(impvol_points,self.gap_with_existing_points,self.gap_between_new_points,self.number_new_points)

        ### Ploting the volsurface points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Extension of the sampled points of the implied volatility surface")
        ax.set_xlabel('Time')
        ax.set_ylabel('Moneyness')
        ax.set_zlabel('Vol')
        ax.scatter(impvol_points[:,0],impvol_points[:,1] ,impvol_points[:,2], cmap='viridis')
        ax.view_init(elev=20, azim=-10)
        plt.show()
        ###

        min_values_impvol = np.min(impvol_points,axis=0)
        max_values_impvol = np.max(impvol_points,axis = 0)
        self.scale = max_values_impvol-min_values_impvol
        self.offset = min_values_impvol

        #normalizing the data for better interpolation
        normalized_impvol_points = (impvol_points-self.offset)/self.scale #I(T,K)

        rbf = Rbf(normalized_impvol_points[:,0], normalized_impvol_points[:,1], normalized_impvol_points[:,2], function='thin_plate',smooth = 0)
        self.thin_plate_spline = rbf

        extrapolation_grid_tenor  = np.linspace(0,1,self.N_t)
        extrapolation_grid_moneyness=np.linspace(0,1,self.N_x)
        extrapolation_grid_tenor,extrapolation_grid_moneyness = np.meshgrid(extrapolation_grid_tenor,extrapolation_grid_moneyness)

        normalized_extrapolated_implied_volatility = self.thin_plate_spline(extrapolation_grid_tenor,extrapolation_grid_moneyness)

        extrapolated_implied_volatility = normalized_extrapolated_implied_volatility*self.scale[2]+self.offset[2] #I(K,T)
        
        ### Ploting implied volatility surface
        fig = plt.figure()
        x1 = np.linspace(min_values_impvol[0],max_values_impvol[0],self.N_t)
        y1 = np.linspace(min_values_impvol[1],max_values_impvol[1],self.N_x)
        x1,y1 = np.meshgrid(x1,y1)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Implied volatility")
        ax.plot_surface(x1, y1,np.clip(extrapolated_implied_volatility,0,1),cmap = "jet")
        ax.set_xlabel('Time')
        ax.set_ylabel('Moneyness')
        ax.set_zlabel('Vol')
        ax.view_init(elev=20, azim=-10)
        plt.show()
        ###

        range_T = [min_values_impvol[0],max_values_impvol[0]]
        range_K = [min_values_impvol[1],max_values_impvol[1]]

        link_formula = local_vol_as_function_of_implied_vol(extrapolated_implied_volatility,range_T,range_K, self.interest_rate, self.initial_spot,self.smoothing)
        local_volatility_array = np.clip(link_formula.local_volatility(),self.min_value_for_local_vol,None)

        ### Ploting the local volatility
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Local volatility")
        ax.plot_surface(x1,y1,np.clip(local_volatility_array,0,1),cmap = 'jet')
        ax.set_xlabel('Time')
        ax.set_ylabel('Moneyness')
        ax.set_zlabel('Vol')
        ax.view_init(elev=20, azim=-10)
        plt.show()
        ###

        X,Y = np.linspace(0,1,self.N_x),np.linspace(0,1,self.N_t)
        interp_function = RegularGridInterpolator((X, Y), local_volatility_array.T)

        self.local_volatility = lambda t,k : interp_function((t,k))

    def get_scale_offset(self):
        return(self.scale, self.offset)

    def __call__(self, *args) -> float:
        """
        Constructs the spline used for interpolation

        Parameters
        ----------
        y : arr(float)
            float array of y-values
        """

        return self.get_vol(args)

    def __str__(self) -> str:
        """
        Prints volatility market_data details

        Returns
        -------
        str
            volatility market_data details - thin plate spline, market_data
        """

        return f"{self.type.capitalize()} Volatility Curve\nmarket_data: {self.market_data}"

    def get_vol(self, *args) -> float:
        """
        Retrieves an interpolated volatility

        Parameters
        ----------
        t : float
            float of t-value
        x : float
            float array of y-values
        """
        t,x = args
        T = (t-self.offset[0])/self.scale[0] #adimensionning
        X = (x-self.offset[1])/(self.scale[1]+2) #adimensionning
        # making sure the obtained values are in [0,1]
        T = np.clip(T,0,1) 
        X_clipped = np.clip(X, 0, 1)
        return np.atleast_1d(self.local_volatility(T,X_clipped))

    def get_market_data(self) -> dict:
        """
        Returns the market_data as a dict

        Returns
        -------
        dict
            A dict with keys equal to the x-array and values euqal to the y-array
        """

        return self.market_data