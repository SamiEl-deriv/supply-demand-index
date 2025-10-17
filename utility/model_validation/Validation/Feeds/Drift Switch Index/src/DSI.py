import numpy as np
import numba as nb
import pandas as pd
from utils import timer, read_yaml, plot_feeds
from alive_progress import alive_bar
from contextlib import nullcontext
# Base specs
DSI_SPECS : dict[str,dict] = read_yaml('./DSI_specs.yaml')
SPREAD_SPECS : dict[str,dict] = read_yaml('./DSI_spread_specs.yaml')
OFFERED = list(DSI_SPECS.keys())
SPREAD_VERSIONS = list(SPREAD_SPECS.keys())

class DSI_Engine():
    """
    Custom DSI generator class

    To use, instantiate with one of production's DSI names, ['DSI10', 'DSI20', 'DSI30']
    And any compatible parameters (specified in yaml file) if necessary

    >>> from DSI import DSI_engine
    >>> DSI_instance = DSI_Engine('DSI_10', x0 = 10_000, mu = 10, ...)

    Then generate paths using one of:

    >>> a = DSI_instance.generate_DSI_index(num_steps=86400, ...)
    >>> b = DSI_instance.generate_DSI_index_info(num_steps=86400, ...)

    For Monte Carlo Simulation, use the following:
    
    >>> cs = DSI_instance.generate_multiple_DSI(iters = 100, num_steps = 86400, ...)
    >>> ds = DSI_instance.generate_multiple_DSI_with_info(iters = 100, num_steps  = 86400)
    
    """

    def __init__(self, base_DSI : str, specs_version, use_prod_gamma = False, **custom_specs) -> None:
        if base_DSI not in OFFERED:
            raise ValueError(f"Invalid DSI Base type.\nSelect one of {OFFERED} instead")
        self.base_type = base_DSI

        DSI_specs = DSI_SPECS[base_DSI]
        if specs_version not in SPREAD_VERSIONS:
            raise ValueError(f"Invalid spread configuration. Try one of {SPREAD_VERSIONS}")
        DSI_spread_specs = SPREAD_SPECS[specs_version][base_DSI]
        DSI_specs.update(DSI_spread_specs)
        DSI_specs.update(custom_specs)
        

        self.x0 = DSI_specs['x0']
        self.mu = DSI_specs['mu']
        self.sigma = DSI_specs['sigma']
        self.T = DSI_specs['T']
        self.dt = 1/(365*86400)
        self.use_prod_gamma = use_prod_gamma
        # Calculate 3-ways gamma & state
        self.set_state()

        # Set up spread parameters
        self.perf = DSI_specs['perf']
        self.ask_epsilon = DSI_specs['ask_epsilon']
        self.ask_kappa = DSI_specs['ask_kappa']
        self.bid_epsilon = DSI_specs['bid_epsilon']
        self.bid_kappa = DSI_specs['bid_kappa']
        self.com_min = DSI_specs['com_min']
        self.com_max = DSI_specs['com_max']
        self.pips = DSI_specs['pips']

    def set_specs(self, **specs):
        """
        Set specified variables.
        If mu, dt or T are modified, set_gamma_state is run to update the state matrix
        
        Arguments
        ----------
        x0 : float
            Starting spot price
        mu : float
            Mean multiplier
        sigma : float
            Constant volatility
        T : float
            Characteristic Time (Expected time to stay in a regime)
        dt : float
            Timestep
        state : np.ndarray (3,3)
            A 3x3 matrix representing the Markov transition matrix
        """
        for key, value in specs.items():
            setattr(self, key, value)
        
        if {'mu', 'dt', 'T'}.difference(specs.keys()) != {'mu', 'dt', 'T'}:
            self.set_state()

    @staticmethod
    def get_DSI_base_specs(base_DSI : str):
        """
        Retrieves DSI production specifications from DSI_specs.yaml

        Arguments
        ---------
        base_DSI : str
            String representing a Production impelemnted DSI

        Returns
        -------
        dict
            The specified DSI specs
        """
        if base_DSI not in OFFERED:
            raise ValueError(f"Invalid DSI Base type.\nSelect one of {OFFERED} instead")
        return DSI_SPECS[base_DSI]
    
    @staticmethod
    def get_DSI_base_spread_specs(base_DSI : str, spread_version : str):
        """
        Retrieves DSI production specifications from DSI_spread_specs.yaml

        Arguments
        ---------
        base_DSI : str
            String representing a Production impelemented DSI
        spread_version : str
            String representing the spread specs version to use

        Returns
        -------
        dict
            The specified DSI specs
        """
        if base_DSI not in OFFERED:
            raise ValueError(f"Invalid DSI Base type.\nSelect one of {OFFERED} instead")
        if spread_version not in SPREAD_VERSIONS:
            raise ValueError(f"Invalid spread configuration. Try one of {SPREAD_VERSIONS}")
        return SPREAD_SPECS[spread_version][base_DSI]
    
    def set_state(self, gamma : float = None):
        """
        Changes the state matrix with a given gamma. 

        Arguments
        ---------
        gamma : float | None
            How likely the stationary regime is to move into the positive regime
            Set gamma = None to reset state matrix:
            * If use_prod_gamma is True, then use the production gamma from DSI_specs.yaml
            * If not, calculate gamma using current parameters

        """
        # Calculate 3-ways gamma & state
        lambda_val = 1 / self.T
        if gamma is None:
            if self.use_prod_gamma:
                gamma = DSI_SPECS[self.base_type]['gamma']
            else:
                gamma = self.calculate_gamma(lambda_val)
        self.state = self.calculate_state(lambda_val, gamma)

    def calculate_gamma(self, lambda_val : float):
        """
        Calculates the probability the stationary regime is to move into the positive regime such that the DSI is driftless

        Arguments
        ---------
        lambda_val : float
            The probability of a state change every tick (no gamma involved)

        Returns
        -------
        float
            Gamma: the probability the stationary regime is to move into the positive regime such that the DSI is driftless
        """
        xp = (lambda_val * np.exp(self.mu * self.dt))/(1 - np.exp(self.mu * self.dt)*(1-lambda_val)) - 1
        xm = (lambda_val * np.exp(-self.mu * self.dt))/(1 - np.exp(-self.mu * self.dt)*(1-lambda_val)) - 1
        return (-2 * xm - xp * (1+xm))/(xp - xm)
    
    @staticmethod
    def calculate_state(lambda_val : float, gamma : float):
        """
        Calculates the Markov state matrix
        
        Arguments
        ---------
        lambda_val : float
            The probability of a state change every tick (no gamma involved)
        gamma : float
            How likely the stationary regime is to move into the positive regime

        Returns
        -------
        numpy.ndarray
            The 3x3 Markov transition matrix corresponding to the regime process
        """
        return np.array([
        [ 1 - lambda_val    , gamma * lambda_val    , lambda_val / 2],
        [lambda_val / 2     , 1 - lambda_val        , lambda_val / 2],
        [lambda_val / 2     , (1-gamma) * lambda_val, 1 - lambda_val]
        ])

    def generate_DSI_spot(self, rng : np.random.Generator, num_steps : int = 86400, extra_info=False):
        """
        Generates DSI spot prices

        Arguments
        ---------
        rng : numpy.random.Generator
            A Numpy random number generator class
        num_steps : int
            The number of timesteps to generate
        extra_info : bool
            If True, returns state information and the feed without noise
        
        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray) |   
            ARRAYS containing
            * Spot price
            * Log returns
            * Spot price without noise (OPTIONAL)
            * State process (OPTIONAL)
        """
        # Generate random numbers outside the function
        random_uniform = rng.random(size=num_steps)
        alpha = _generate_alpha(self.state, random_uniform)
        alpha_mu = (1-alpha) * self.mu

        random_normal = rng.normal(size=num_steps)
        ret = np.zeros(num_steps+1)
        ret[0] = 0
        ret[1:] = (alpha_mu[1:] - self.sigma **2 / 2) * self.dt + self.sigma * np.sqrt(self.dt)*random_normal
        
        DSI_spot = self.x0*np.cumprod(np.exp(ret))
        log_returns = np.log(DSI_spot[1:] / DSI_spot[:-1])

        if extra_info:
            drifty = np.exp(alpha_mu * self.dt)
            drift_sim = self.x0*np.cumprod(drifty)
            return DSI_spot, log_returns, drift_sim, alpha
        else:
            return DSI_spot, log_returns

    def generate_spread(self, DSI_spot, log_returns, extra_info=False):
        """
        Generates the bid and ask prices from diff

        Arguments
        ---------
        DSI_spot : numpy.ndarray
            The spot prices
        log_returns : numpy.ndarray
            The log returns
        extra_info : bool
            If True, returns diff as well
        
        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) |   
            Arrays containing
            * Ask price
            * Bid price
            * diff (OPTIONAL)
        """

        # Generate diff values
        _, _, diff = _generate_diff(self.mu, self.sigma, self.dt, self.state, log_returns)

        # Convert diff values to actual spread
        fair_spread_ask, fair_spread_bid, com_ask, com_bid = _spread(self.perf, 
                                                                     self.ask_kappa, self.ask_epsilon, 
                                                                     self.bid_kappa, self.bid_epsilon, 
                                                                     self.pips, self.com_max, self.com_min, 
                                                                     np.array(diff))

        # To ensure bid/ask prices at t=0 are equal to spot prices
        ask_price = DSI_spot * (1 + fair_spread_ask) + com_ask
        bid_price = DSI_spot * (1 - fair_spread_bid) - com_bid

        if extra_info:
            return ask_price, bid_price, diff
        else:
            return ask_price, bid_price
        
    def generate_DSI_index(self, num_steps=86400, pandas=False, seed=None, timed=False):
        """
        Generates simulated DSI

        Arguments
        ---------
        state : np.ndarray
            A 3 x 3 array representing the underlying Markov state transitition matrix
        num_steps : int
            The number of iterations to make (in seconds)
        pandas : bool
            If true, return pandas dataframe. Otherwise return numpy ndarrays
        seed : int
            The seed for np.random

        Returns
        -------
        pandas.DataFrame | tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) |   
            A dataframe/arrays containing
            * Spot price
            * Bid price
            * Ask price
        """

        # Set seed locally
        rng = np.random.default_rng(seed)
        if timed:
            DSI_generator = timer(subject=f'{self.base_type} Spot Generation')(self.generate_DSI_spot)
            spread_generator = timer(subject=f'{self.base_type} Spread Generation')(self.generate_spread)
        else:
            DSI_generator = self.generate_DSI_spot
            spread_generator = self.generate_spread
        DSI_spot, log_returns = DSI_generator(rng, num_steps = num_steps)
        ask_price, bid_price = spread_generator(DSI_spot, log_returns)
        if pandas:
            return pd.DataFrame.from_dict({'bid' : bid_price, 'spot' : DSI_spot, 'ask' : ask_price})
        else:
            return bid_price, DSI_spot, ask_price

    def generate_DSI_index_info(self, num_steps=86400, seed=None, timed=False):
        """
        Generates simulated DSI with extra information

        Arguments
        ---------
        state : numpy.ndarray
            A 3 x 3 array representing the underlying Markov state transitition matrix
        num_steps : int
            The number of timesteps to generate
        seed : int
            The seed for np.random

        Returns
        -------
        pandas.DataFrame   
            A dataframe containing
            * Spot price
            * Bid price
            * Ask price
            * Drift
            * Alpha (states)
            * diff
        """
        # Set seed locally
        rng = np.random.default_rng(seed)
        if timed:
            DSI_generator = timer(subject=f'{self.base_type} Spot Generation')(self.generate_DSI_spot)
            spread_generator = timer(subject=f'{self.base_type} Spread Generation')(self.generate_spread)
        else:
            DSI_generator = self.generate_DSI_spot
            spread_generator = self.generate_spread
        DSI_spot, log_returns, drift_sim, alpha = DSI_generator(rng, num_steps = num_steps, extra_info = True)
        ask_price, bid_price, diff = spread_generator(DSI_spot, log_returns, extra_info = True)
        return pd.DataFrame.from_dict({'spot' : DSI_spot,'bid' : bid_price, 'ask' : ask_price, 'log_returns' : np.insert(log_returns, 0, np.nan), 'drift' : drift_sim, 
                                    'state' : alpha, 'diff' : diff})
    
    def generate_multiple_DSI(self, iters : int = 100, num_steps : int = 86400, pandas=False, verbose=False):
        """
        Naive Monte Carlo simulation of DSI

        Arguments
        ---------
        iters : int
            Number of DSI to generate
        num_steps : int
            The number of timesteps to generate
        pandas : bool
            If True, return a pandas dataframe

        Returns
        -------
        np.ndarray(np.ndarray,...)
            A Numpy array containing numpy arrays containing
            * Spot price
            * Bid price
            * Ask price
        """
        result = []
        with alive_bar(iters, bar='filling', spinner='radioactive') if verbose else nullcontext() as bar:
            for _ in range(iters):
                result.append(self.generate_DSI_index(num_steps, pandas))
                if bar is not None:
                    bar()
        return result
    
    def generate_multiple_DSI_with_info(self, iters : int = 100, num_steps : int = 86400, verbose = False):
        """
        Naive Monte Carlo simulation of DSI with additional information

        Arguments
        ---------
        iters : int
            Number of DSI to generate
        num_steps : int
            The number of timesteps to generate

        Returns
        -------
        list[pandas.DataFrame]
            A list of dataframes containing
            * Spot price
            * Bid price
            * Ask price
        """
        result = []
        with alive_bar(iters, bar='filling', spinner='radioactive') if verbose else nullcontext() as bar:
            for _ in range(iters):
                result.append(self.generate_DSI_index_info(num_steps))
                if bar is not None:
                    bar()
        return result

    
    
@nb.njit(fastmath=True)
def norm_pdf(x : np.ndarray, mu, sigma):
    """
    Normal probability density function - WS Optimized

    Arguments
    ---------
    x : numpy.ndarray
        The log-return samples
    mu : np.ndarray
        The drifts for each regime
    sigma : float
        Constant volatility

    Returns
    -------
    numpy.ndarray
        Probability mass per log return per drift
    """
    result_pdf = np.empty((x.size, mu.size))
    for i in nb.prange(x.size):
        for j in nb.prange(mu.size):
            result_pdf[i, j] = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-((x[i] - mu[j]) / sigma)**2)
    return result_pdf

@nb.njit(fastmath = True)
def _spread(perf, ask_kappa, ask_epsilon, bid_kappa, bid_epsilon, pip, com_max, com_min, x):
    """
    Current iteration of spread function, circa 10/08/2023

    Arguments
    ---------
    ask_kappa : float
        The expected profit of a perfect long-short strategy in the positive regime
    ask_epsilon : float
        The threshold for when ask spread should be applied
    bid_kappa : float
        The expected profit of a perfect long-short strategy in the negative regime
    bid_epsilon : float
        The threshold for when bid spread should be applied
    pip : float
        The pip size
    com_max : float
        The maximum commission possible (at diff = ±1)
    com_min : float
        The minimum commission possible (at diff = 0)
    x : numpy.ndarray  
        Diff values

    Returns
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        A tuple of numpy arrays containing
        * Ask spread
        * Bid spread
        * Ask commission
        * Bid commission
    """
    fair_spread_ask = np.maximum(perf * ask_kappa * x + ask_epsilon,  0)
    fair_spread_bid = np.maximum(-perf * bid_kappa * x + bid_epsilon, 0)
    com_ask = pip * np.maximum((com_max - com_min) * x + com_min, 0) / 2
    com_bid = pip * np.maximum((com_min - com_max) * x + com_min, 0) / 2
    return fair_spread_ask, fair_spread_bid, com_ask, com_bid

@nb.njit()
def _generate_alpha(state, random_nums):
    """
    Generates the regimes/states via a Markov sampling process

    Arguments
    ---------
    state : numpy.ndarray
        3x3 Numpy array representing the Markov transition matrix
    random_nums : numpy.ndarray
        Uniform random numbers

    Returns
    -------
    numpy.ndarray
        The array of states satisfying the Markov process (size = len(random_nums)+1)
    """
    alpha = np.empty(len(random_nums)+1, dtype=np.int64)
    alpha[0] = 1
    for i in range(0, len(random_nums)):
        state_1 = int(random_nums[i] >= state[0,alpha[i]])
        state_2 = int(random_nums[i] >= state[0,alpha[i]] + state[1,alpha[i]])
        alpha[i+1] = state_1 + state_2
    return alpha

# Generate diff algorithm values -- WS Sped up
@nb.njit(fastmath=True)
def _generate_diff(mu, sigma, dt, state, log_returns, xi_0 = np.array([0,1,0], dtype=np.float64)):
    """
    Generates diff values for spread pricing - WS optimized

    Arguments
    ---------
    mu : float
        Mean multiplier
    sigma : float
        Constant volatility
    dt : float
        Timestep
    state : numpy.ndarray
        3x3 Numpy array representing the Markov transition matrix
    log_returns : numpy.ndarray
        Array of log returns
    xi_0 : numpy.ndarray, Default [0,1,0]
        Starting distribution

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Numpy arrays contatining
        * Inference probabilities
        * Forecasting probabilities
        * diff values
    """
    num_steps = log_returns.shape[0]
    mus = np.array([mu, 0, -mu], dtype=np.float64)
    xi_forecast = np.zeros((num_steps+1, 3), dtype=np.float64)
    xi_forecast[0] = xi_0
    xi_inference = np.zeros((num_steps, 3), dtype=np.float64)
    mumu = (( mus - sigma ** 2 / 2) * dt)
    conditional_dists = norm_pdf(log_returns, mu=mumu, sigma = sigma * np.sqrt(dt))
    for i in range(num_steps):
        deno = np.dot(xi_forecast[i,:], conditional_dists[i,:])
        for j in range(3):
            xi_inference[i,j] = (xi_forecast[i, j] * conditional_dists[i, j]) / deno
        xi_forecast[i+1] = state @ xi_inference[i]
    diff = xi_forecast[:, 0] - xi_forecast[:,2]
    return xi_inference, xi_forecast, diff


class DSI_Factory():
    """
    Quick and dirty factory class
    """

    @staticmethod
    def create_DSI(base_DSI : str, specs_version : str, use_prod_gamma = False, **custom_specs):
        return DSI_Engine(base_DSI, specs_version, use_prod_gamma, **custom_specs)



if __name__ == "__main__":

    # iters = 1
    # num_steps = 86400 * 30
    # print(num_steps)
    # DSI = DSI_Engine('DSI10')
    # transition = DSI.state
    
    # final_states = np.empty(iters)
    # for i in range(iters):
    #     random_nums = np.random.random(size=(num_steps))
    #     states = _generate_alpha(transition, random_nums)
    #     print(states[2592000])
    #     final_states[i] = states[-1]
    
    DSI_choice = 'DSI20'
    SPREAD_VERSION = 'Demo_01092023'
    DSI = DSI_Engine(DSI_choice, SPREAD_VERSION)
    asd = DSI.generate_DSI_index_info(timed=True)
    print(asd)
    # paths = DSI.generate_multiple_DSI(iters=100)
    # start_index = 0
    # end_index = 86400

    # plot_feeds(asd, (['spot', 'bid', 'ask'], ['state', 'diff']), (['Spot', 'Bid', 'Ask'], ['State', 'diff']), ('price', ''), 
    #        f'{DSI_choice} Simulation', start_index, end_index)
