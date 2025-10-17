import numpy as np
import scipy.stats as ss
import scipy.linalg as spla
from typing import Callable
from ...options.payoff.payoff import PayOff
from .returns_path_generators.RPG_base import ReturnPathGenerator
from ...market.stochastic_indices.stochastic_index_base import StochasticIndex, dt


class MonteCarloEngine:
    """
    The Monte Carlo engine class. 

    It is responsible of the drawing path of value, starting from a given
    initial spot and finishing at a given maturity, of a given index according
    to the logic of the given paths genearor object and the stochastic index
    itself. For every paths the payoff, according to the given option payOff
    object, is computed. The price of this option is then get by averaging all
    of these payoffs. This averaging can be weighted according to the type of
    Monte Carlo variance reduction method.
    """

    def __init__(self,
                 index: StochasticIndex,
                 option_PO: PayOff,
                 time_to_expiry: float,
                 S_0: float,
                 N: int,
                 discount_rate: float = 0,
                 dt: float = dt,
                 RPG: type[ReturnPathGenerator] = None,
                 seed: int = None) -> None:
        """
        Parameters
        ----------
        index : StochasticIndex
            The stochastic process of the market concerned by the option
            to be priced.
        option_PO : PayOff
            The payoff object of the option we want to price.
        time_to_expiry : float
            The maturity of the option (in terms of years).
        S_0 : float
            The initial/actual spot of the market at which the option is
            priced.
        N : int
            The number of paths to be drawned.
        dt : float
            The time lapse between two ticks in the market (in terms of years).
            Default is set to be one second.
        paths_gen : PathsGenerator
            The paths generator used to build the paths of index values.
            Default is set to StandardPathsGenerator
        discount_rate_curve : YieldCurve
            The discount rate of the currency used to pay the option. 
        seed : int
            The seed of the randum number generator. 
        """

        # Initialize simulation seed
        self.__seed = seed  # The original unmutable seed

        # Extract Options parameters
        self.__expiry = time_to_expiry
        self.__option = option_PO
        self.__index = index
        self.__discount_rate = discount_rate

        # Extract paths parameter
        self.__paths_params0 = {'S_0': S_0,
                                'T': time_to_expiry,
                                'N': N,
                                'dt': dt,
                                'seed': seed}
        self.__paths_params = self.__paths_params0.copy()

        # Setting up the returns paths drawing method
        if not RPG:
            self.__RPG = RPG(self.__index)
            self.__RPG.replaceDrawingMethod()
        
        #Drawing stochastic paths
        self.__paths0 = self.__index.drawValuePath(S_0, time_to_expiry,
                                                   N, dt, seed)
        self.__paths = self.__paths0

        # Set the default equal weight for each samples drawn
        self.__weights = 1/N

    def redraw(self, **new_params) -> None:
        """
        Method that is used to change some parameters of the simulations.
        The parameters that can be changed are: initial spot, time to expiry,
        number of paths, ticks timelapse and the seed of the random number
        generator. The simulation is then redrawned according to new values of 
        parameters.

        Parameters
        ----------
        **new_params : 
            Parameters of the new simulation. This parameters can be choosed 
            within: {S_0: float, T: float, N: int, dt: float, seed: int}
        """
        # Test if new_params keys are well defined
        if not set(new_params.keys()).issubset(set(self.__paths_params0.keys())):
            raise KeyError("key words arguments of 'redraw' method should be"
                           f" in {set(self.__paths_params0.keys())}")

        # Copy (shallow) dict of default params for paths
        self.__paths_params = self.__paths_params0.copy()

        # Replace modified params by their new values
        for key in new_params.keys():
            self.__paths_params[key] = new_params[key]

        # Redraw the paths
        self.__paths = self.__index.drawValuePath(self.__paths_params['S_0'],
                                                  self.__paths_params['T'],
                                                  self.__paths_params['N'],
                                                  self.__paths_params['dt'],
                                                  self.__paths_params['seed'])

        # Ensure that the weights are reset to be uniform.
        self.set_weights(1/self.__paths_params['N'])

    def reset(self) -> None:
        """
        Reset the simulation to the paths generated at the initiation of the 
        instance as well as its corresponding parameters.
        """
        self.__paths_params = self.__paths_params0
        self.__paths = self.__paths0

    @property
    def refSpot(self) -> float:
        """
        Retrieve the initial spot

        Returns
        -------
        float
            The initial spot
        """
        return self.__paths_params['S_0']

    @property
    def time_resolution(self) -> float:
        """
        Retrieve the ticks timelapse

        Returns
        -------
        float
            The ticks timelapse
        """
        return self.__paths_params['dt']

    @property
    def maturity(self) -> float:
        """
        Retrieve the time to expiry of the option

        Returns
        -------
        float
            The time to expiry
        """
        return self.__paths_params['T']

    @property
    def discount(self) -> float:
        """
        Retrieve the discount

        Returns
        -------
        float
            The discount factor: e^(-r*T)
        """
        return np.exp(-self.__discount_rate*self.__paths_params['T'])

    @property
    def numOfPaths(self) -> float:
        """
        Retrieve the number of paths

        Returns
        -------
        int
            The number of paths
        """
        return self.__paths_params['N']

    @property
    def seed(self) -> float:
        """
        Retrieve the seed of the random number generator

        Returns
        -------
        int
            The seed of the random number generator
        """
        return self.__paths_params['seed']

    @property
    def paths(self) -> np.ndarray:
        """
        Retrieve the simulated paths.

        Returns
        -------
        array
            The simulated paths
        """
        return self.__paths

    @property
    def payOffType(self) -> str:
        """
        Retrieve the name of the option

        Returns
        -------
        str
            The name of the option
        """
        return self.__option.__name__

    @property
    def index(self) -> str:
        """
        Retrieve the ID of the index

        Returns
        -------
        str
            The ID of the index
        """
        return self.__index.__repr__()

    @property
    def weights(self) -> np.ndarray:
        """
        Retrieve the weights of each simulated paths

        Returns
        -------
        array
            The weights of each simulated paths
        """
        return self.__weights

    def set_weights(self, weights) -> None:
        """
        Set the weights for each paths in the payOff averaging
        for computing the price.

        Parameters
        ----------
        weights: array of floats
            The array of the weights. If the len of the array
            does not match the number of paths, a exception is
            raised. If it is a single value, this is the same things
            as setting equal weights.
        """
        try:
            new_weights = weights * np.ones(self.numOfPaths)
        except ValueError:
            raise ValueError("Number of new weights should equal number"
                             "of path samples drawn!")

        # Renormalize if needed
        self.__weights = new_weights / np.sum(new_weights) / self.numOfPaths

    @property
    def payOffOnPath(self) -> Callable:
        """
        Retrieve the payoff on path function

        Returns
        -------
        callable
            The 'onPath' method of the payOff object.
        """
        return self.__option.onPath

    @property
    def resulting_payOffs(self) -> np.ndarray:
        """
        Retrieve the corresponding payoff for each paths drawn.

        Returns
        -------
        array of float
            The array of payoff for each paths.
        """
        return self.payOffOnPath(self.__paths)

    def get_price(self) -> float:
        """
        Get the price of the option computed from the (weighted)
        average of the payoffs of the option computed for each
        simulated paths.

        Returns
        -------
        float
            The estimated price of the option.
        """
        return np.sum(self.weights * self.resulting_payOffs)

    def get_std(self) -> float:
        """
        Get the standard deviation associated to the simulation.

        Returns
        -------
        float
            the estimated standard deviation of the payoffs.
        """
        return np.std(self.weights * self.resulting_payOffs * self.numOfPaths)

    def get_confidence_interval(self, delta: float) -> np.ndarray:
        """
        Get the confidence interval of the estimated price. This is computed
        using the generic formula given by the law of large numbers.

        NOTE: It is not valid for quasi monte carlo simulation.

        Parameters
        ----------
        delta: float
            The percentage of error.

        Returns
        -------
        tuple of floats
            the estimated interval of confidence.
        """
        sigma = self.get_std()
        error_quantile = np.array(ss.norm.interval(1-delta/2))
        abs_error = error_quantile*sigma/np.sqrt(self.numOfPaths)
        return self.get_price() + abs_error
