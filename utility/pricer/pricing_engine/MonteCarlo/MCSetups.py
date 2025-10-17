import numpy as np
import scipy.linalg as spla
from .MCEngine import MonteCarloEngine
from ...options.payoff.payoff import PayOff
from .returns_path_generators.RPG_base import ReturnPathGenerator
from ...market.stochastic_indices.stochastic_index_base import StochasticIndex, dt


def MCstdSetup(index: StochasticIndex,
               option_PO: PayOff,
               time_to_expiry: float,
               S_0: float,
               N: int,
               dt: float = dt,
               RPG: type[ReturnPathGenerator] = None,
               discount_rate: float = 0.0,
               seed: int = None) -> MonteCarloEngine:
    """
    Instantiate a MonteCarloEngine object

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
    seed : int
        The seed of the randum number generator. 

    Returns
    -------
    MonteCarloEngine
        the MonteCarloEngine instance.
    """
    return MonteCarloEngine(index, option_PO, time_to_expiry, S_0, N, dt=dt,
                            RPG=RPG, discount_rate=discount_rate,
                            seed=seed)


def MCunderlyingCVSetup(index: StochasticIndex,
                        option_PO: PayOff,
                        time_to_expiry: float,
                        S_0: float,
                        N: int,
                        dt: float = dt,
                        RPG: type[ReturnPathGenerator] = None,
                        discount_rate: float = 0,
                        seed: int = None) -> MonteCarloEngine:
    """
    Instantiate a MonteCarloEngine object and set the correct weights
    according to a control variate variance reduction technique where the
    underlying spot paths is the control variates.

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
    seed : int
        The seed of the randum number generator. 

    Returns
    -------
    MonteCarloEngine
        the MonteCarloEngine instance with weights setup as control
        variates technique using the underlying spot paths.
    """
    MCEngine = MonteCarloEngine(index, option_PO, time_to_expiry, S_0, N, dt=dt,
                                RPG=RPG, discount_rate=discount_rate,
                                seed=seed)
    X = MCEngine.paths
    N = X.shape[0]
    Xbar = np.array([np.mean(X, axis=0)])
    S_X = (X.T @ X - N * Xbar.T @ Xbar) / (N - 1)
    S_Xinv = spla.inv(S_X)
    E_X = index.forward(S_0, time_to_expiry)
    weigths = np.array([(N-1)/N + (Xbar - X[i, :]) @ S_Xinv @ (Xbar - E_X).T
                        for i in range(N)])
    MCEngine.set_weights(weigths)
    return MCEngine


def MCoptionCVSetup(index: StochasticIndex,
                    option_PO: PayOff,
                    time_to_expiry: float,
                    S_0: float,
                    N: int,
                    optionCV_PO: PayOff,
                    optionCV_price: float,
                    dt: float = dt,
                    RPG: type[ReturnPathGenerator] = None,
                    discount_rate: float = 0,
                    seed: int = None) -> MonteCarloEngine:
    """
    Instantiate a MonteCarloEngine object and set the correct weights
    according to a control variate variance reduction technique where the
    control variates is another option payoff which price is known.

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
    optionCV_PO : PayOff
        A payOff instance corresponding to the control variate option payoff.
    optionCV_price : float
        The price of the option which payoff is used as a control variate.
    dt : float
        The time lapse between two ticks in the market (in terms of years).
        Default is set to be one second.
    paths_gen : PathsGenerator
        The paths generator used to build the paths of index values.
        Default is set to StandardPathsGenerator
    seed : int
        The seed of the randum number generator. 

    Returns
    -------
    MonteCarloEngine
        the MonteCarloEngine instance with weights setup as control
        variates technique using the given other option payoff.
    """
    MCEngine = MonteCarloEngine(index, option_PO, time_to_expiry, S_0, N, dt=dt,
                                RPG=RPG, discount_rate=discount_rate,
                                seed=seed)
    X = optionCV_PO.onPath(MCEngine.paths)
    N = X.shape[0]
    Xbar = np.mean(X)
    weigths = 1 + (Xbar - X) * (Xbar - optionCV_price) / np.std(X)**2
    MCEngine.set_weights(weigths)
    return MCEngine


MCSetupsRefsDict: dict[str, callable] = {
    "standard"     : MCstdSetup,
    "UnderlyingCV" : MCunderlyingCVSetup,
    "OptionCV"     : MCoptionCVSetup
}
