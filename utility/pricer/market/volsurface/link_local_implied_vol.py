import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

class local_vol_as_function_of_implied_vol:
    """A class that givenan implied volatility surface computes the corresponding 
    local volatility surface via the link formula."""

    def __init__(self,implied_volatility : np.ndarray,range_T: np.ndarray,range_K: np.ndarray,r : float,S0 : float, smoothing : float) -> None:
        self.implied_volatility = implied_volatility
        n,p = np.shape(self.implied_volatility)
        self.N_K = n
        self.N_T = p
        self.dT = (range_T[1]-range_T[0])/self.N_T
        self.dK = (range_K[1]-range_K[0])/self.N_K
        self.interest_rate =r
        self.initial_spot = S0
        self.array_T = np.tile(np.linspace(range_T[0],range_T[1],self.N_T), (self.N_K, 1))
        self.array_K = np.tile(np.linspace(range_K[0],range_K[1],self.N_K), (self.N_T, 1)).T

        #Computing the partial derivatives with respect to T on the original surface
        _, partial_T = np.gradient(self.implied_volatility,self.dK,self.dT)

        # Smoothing the implied volatility surface in the moneyness direction
        linspace_moneyness = np.linspace(range_K[0],range_K[1],self.N_K)
        smoothed_impvol = np.copy(implied_volatility)
        for tenor in range(self.N_T):
            volsmile = self.implied_volatility[:,tenor]
            cs = UnivariateSpline(linspace_moneyness, volsmile, k=3, s=smoothing)
            new_volsmile = cs(linspace_moneyness)
            smoothed_impvol[tenor]=new_volsmile

        # Displaying the smoothed surface : 
        fig = plt.figure()
        linspace_time = np.linspace(range_T[0],range_T[1],self.N_T)
        linspace_time,linspace_moneyness = np.meshgrid(linspace_time,linspace_moneyness)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("smoothed implied vol'")
        ax.plot_surface(linspace_time,linspace_moneyness,smoothed_impvol.T,cmap = 'jet')
        ax.set_xlabel('Time')
        ax.set_ylabel('Moneyness')
        ax.set_zlabel('Vol')
        ax.view_init(elev=20, azim=-10)
        plt.show()
        
        #Computing the partials derivatives with respect to K, on the smoothed surface
        partial_K,_ = np.gradient(smoothed_impvol.T,self.dK,self.dT)
        partial_K2,_ = np.gradient(partial_K,self.dK,self.dT)

        self.partial_T = partial_T
        self.partial_K = partial_K
        self.partial_K2 = partial_K2
        self.discounted = self.array_K*np.exp(-self.interest_rate*self.array_T)
        self.y = np.log(self.array_K/self.initial_spot)-self.interest_rate*self.array_T

    def numerator(self)->np.ndarray:
        return self.implied_volatility**2+ 2*self.implied_volatility*self.array_T*self.numerator_term()
    
    def numerator_term(self)->np.ndarray:
        return self.partial_T+self.interest_rate*self.array_K*self.partial_K
    
    def denominator(self)->np.ndarray:
        return self.denominator_term_1() +self.array_K*self.implied_volatility*self.array_T*self.denominator_term_2()

    def denominator_term_1(self)->np.ndarray:
        return (np.ones((self.N_K,self.N_T))-self.array_K*self.y*self.partial_K/self.implied_volatility)**2
    
    def denominator_term_2(self)->np.ndarray:
        return self.partial_K-0.25*self.array_K*self.implied_volatility*self.array_T*self.partial_K**2 +self.array_K*self.partial_K2

    def local_variance(self)->np.ndarray:
        return np.clip(self.numerator()/self.denominator(),0,None)
    
    def naive_local_variance(self)->np.ndarray:
        return np.clip(self.implied_volatility**2 +2*self.implied_volatility*self.array_T*self.partial_T,0,None)
    
    def local_volatility(self)->np.ndarray:
        return np.sqrt(self.local_variance())
    
    def naive_local_volatility(self)->np.ndarray:
        return np.sqrt(self.naive_local_variance())