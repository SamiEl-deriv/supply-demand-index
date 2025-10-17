from enum import Enum
import numpy as np
from scipy import sparse
from functools import partial


class CDScheme(Enum):
    """
    Enumeration class for usable finite difference schemes

    Attributes
    ----------

    EXPLICIT : The Backwards Time Centered Scheme (BTCS)
    IMPLICIT : The Forwards Time Centered Scheme (FTCS)
    CRANK_NICOLSON : The average of the BTCS and FTCS scheme (special case of the theta method with theta = 0.5)
    """
    EXPLICIT = 1
    IMPLICIT = 2
    CRANK_NICOLSON = 3


class VanillaPayoff:
    """
    A class describing the payoff for a vanilla option

    Attributes
    ----------
    K : float
        Strike price
    call : bool
        If true, then the option is a call option, otherwise it is a put option
    """

    def __init__(self, K, call) -> None:
        """
        Parameters
        ----------
        K : float
            Strike price
        call : bool
            If true, then the option is a call option, otherwise it is a put option
        """

        self.K = K
        self.call = call

    def __call__(self, S) -> float:
        """
        Wrapper for get_payoff

        Parameters
        ----------
        S : float
            Spot price

        Returns
        -------
        float
            Payoff (max(S-K,0) if call else max(K-S,0))
        """

        return self.get_payoff(S)

    def get_payoff(self, S) -> float:
        """
        Returns the payoff given a spot level. Depends on the parity of the option

        Parameters
        ----------
        S : float
            Spot price

        Returns
        -------
        float
            Payoff (max(S-K,0) if call else max(K-S,0))
        """

        phi = 1 if self.call else -1
        return np.maximum(phi * (S - self.K), 0)

    def S_boundary_upper(self, S_max, r, T, t) -> float:
        """
        The upper spot boundary conditions (S=S_max)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        S_max : float
            The maximum spot price
        t : float
            The current time
        T : float
            The time to maturity

        Returns
        -------
        float
            The price of the option at the upper spot boundary
        """
        return S_max - self.K * np.exp(-r * (T - t)) if self.call else 0

    def S_boundary_lower(self, S_min, r, T, t) -> float:
        """
        The lower spot boundary conditions (S=S_min < K)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        S_max : float
            The maximum spot price
        t : float
            The current time
        T : float
            The time to maturity

        Returns
        -------
        float
            The price of the option at the lower spot boundary
        """
        return 0 if self.call else (self.K - S_min) * np.exp(-r * (T - t))


class DigitalPayoff(VanillaPayoff):
    """
    A class describing the payoff for a cash-or-nothing digital option

    Attributes
    ----------
    K : float
        Strike price
    call : bool
        If true, then the option is a call option, otherwise it is a put option
    equal : bool
        If true, then payoff can be received if S = K
    """

    def __init__(self, K, call, equal) -> None:
        super().__init__(K, call)
        self.equal = equal

    def get_payoff(self, S) -> float:
        """
        Returns the payoff given a spot level. Depends on the parity of the option

        Parameters
        ----------
        S : float
            Spot price

        Returns
        -------
        float
            Payoff (1 if in the money or 0 if out of the money)
        """

        payoff_condition = (S > self.K) if self.call else (S < self.K)
        equal_condition = (S == self.K) if self.equal else False

        return (np.logical_or(payoff_condition, equal_condition)).astype(int)

    def S_boundary_upper(self, S_max, r, T, t) -> float:
        """
        The upper spot boundary conditions (S=S_max > K)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        S_max : float
            The maximum spot price
        t : float
            The current time
        T : float
            The time to maturity

        Returns
        -------
        float
            The price of the option at the upper spot boundary
        """
        return np.exp(-r * (T - t)) if self.call else 0

    def S_boundary_lower(self, S_min, r, T, t) -> float:
        """
        The lower spot boundary conditions (S=S_min < K)
        TODO: Implement Neumann Boundary conditions (null-gamma/delta)

        Parameters
        ----------
        S_max : float
            The maximum spot price
        t : float
            The current time
        T : float
            The time to maturity

        Returns
        -------
        float
            The price of the option at the lower spot boundary
        """
        return 0 if self.call else np.exp(-r * (T - t))


class Option:
    """
    A class containing the parameters of a basic option

    Attributes
    ----------
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Dividend rate
    vol : float
        Flat volatility
    payoff : VanillaPayoff
        The payoff structure of the option
    T : float
        Time to maturity
    american : bool
        If true, the option is an American option, otherwise it is a European option
    """

    def __init__(
            self,
            K,
            r,
            vol,
            T,
            payoffclass: VanillaPayoff,
            q=0,
            american=False,
            **payoff_args) -> None:
        """
        Parameters
        ----------
        K : float
            Strike price
        r : float
            Risk-free rate
        vol : float
            Flat volatility
        T : float
            Time to maturity
        payoffclass : VanillaPayoff
            The payoff structure of the option
        q : float
            Dividend rate
        american : bool
            If true, the option is an American option, otherwise it is a European option
        payoff_args : kwargs
            Additional payoff arguments
        """

        self.K = K
        self.r = r
        self.q = q
        self.vol = vol
        self.payoff = payoffclass(K, **payoff_args)
        self.T = T
        self.american = american

    def get_parity(self):
        """
        Returns the parity of the option

        Returns
        -------
        bool
            If True, it is a call option, else it is a put option
        """
        return self.payoff.call


class ConvectionDiffusionPricer:
    """
    A class that solves a backwards-time convection-diffusion equation given the coefficients of the pde:
     V_t = a * V_x + b * V_{xx} + cV

    We call the coefficients:

    a : convection term
    b : diffusion term
    c : source term

    To obtain the grid of solutions, use get_grid.

    There is consideration for American options, set american to None if you want a typical PDE solver
    """

    def __init__(
            self,
            convection,
            diffusion,
            source,
            boundary_terminal,
            boundary_xmin,
            boundary_xmax,
            N_x: int,
            N_t: int,
            x_min,
            x_max,
            t_max,
            american=None) -> None:
        """
        Parameters
        ----------
        convection : func(float, float)
            The convection term (a) of the pde. Must take two floats as an argument, representing (x,t)
        diffusion : func(float, float)
            The diffusion term (b) of the pde. Must take two floats as an argument, representing (x,t)
        source : func(float, float)
            The source term (c) of the pde. Must take two floats as an argument, representing (x,t)
        boundary_terminal : func(float)
            The terminal boundary conditions of the pde. Must take a float as an argument, representing x
        boundary_xmin : func(float)
            The lower boundary conditions of the pde. Must take a float as an argument, representing t
        boundary_xmax : func(float)
            The upper boundary conditions of the pde. Must take a float as an argument, representing t
        N_x : int
            The number of points on the x-axis
        N_t : int
            The number of points on the t-axis
        x_min : float
            The minimum x_value
        x_max : float
            The maximum x-value
        t_max : float
            The maximum t-value
        american : func(float) | None
            If none, the PDE solver works as usual. If a function with 1 input is given,
            it will assume that early exercise for American options is possible
        """

        self.convection = convection
        self.diffusion = diffusion
        self.source = source

        self.boundary_terminal = boundary_terminal
        self.boundary_xmin = boundary_xmin
        self.boundary_xmax = boundary_xmax

        self.N_x = N_x
        self.N_t = N_t
        self.x_min = x_min
        self.x_max = x_max
        self.t_max = t_max

        # Initialize scheme and grid
        self.scheme = None
        self.grid = None

        # American replacement condition
        self.american = american

    def setup_grid(self) -> None:
        """
        Initializes spacing, grids and coordinate arrays. Coordinates are in terms of (x,t) and confined in the rectangle [x_min, x_max] x [0, t_max]

        Parameters
        ----------
        dx : float
            distance between points in x-array
        dt : float
            distance between points in t-array
        x_range : np.ndarray
            Discrete x-axis points
        t_range : np.ndarray
            Discrete t-axis points
        grid : np.ndarray
            Discrete grid with coordinates (x, t) in [0, x_range] x [0, t_range]
        """

        self.dx = self.x_max / (self.N_x - 1)
        self.dt = - self.t_max / (self.N_t - 1)

        self.x_range = np.linspace(self.x_min, self.x_max, self.N_x)
        self.t_range = np.linspace(0, self.t_max, self.N_t)

        self.grid = np.zeros((self.N_x, self.N_t))

    def populate_boundary(self):
        """
        Populates boundary of grid

        Currently supports only Dirichlet Conditions
        TODO: Implement Neumann Conditions
        """

        # Assign terminal boundary conditions
        self.grid[:, -1] = self.boundary_terminal(self.x_range)

        # Assign min/max conditions
        self.grid[0, :-1] = self.boundary_xmin(self.t_range[:-1])
        self.grid[-1, :-1] = self.boundary_xmax(self.t_range[:-1])

    def populate_grid(self, scheme=CDScheme.EXPLICIT) -> None:
        """
        Populates grid with solution of the pde.

        Implemented schemes are explicit, implicit and Crank-Nicolson

        Parameters
        ----------
        scheme : CDScheme
            The scheme to use
        """

        self.setup_grid()
        self.populate_boundary()
        if scheme not in CDScheme:
            raise ValueError(
                f"Invalid scheme. Currently supported schemes are {[CDScheme.__members__.keys()]}")

        # Select scheme
        if scheme == CDScheme.EXPLICIT:
            step_marcher = self.explicit
        elif scheme == CDScheme.CRANK_NICOLSON:
            step_marcher = self.crank_nicolson
        else:
            step_marcher = self.implicit

        # Iterate backwards and populate grid
        for i in range(self.t_range.shape[0] - 2, -1, -1):
            self.grid[:, i] = step_marcher(
                t=self.t_range[i], init_values=self.grid[:, i], prev_values=self.grid[:, i + 1])

    def get_grid(self) -> np.ndarray:
        """
        Retrieves a populated grid - solution to the pde

        Returns
        -------
        numpy.ndarray
            The solution grid if it has been populated
        """
        if self.grid is not None:
            return self.grid
        else:
            raise ValueError("Grid has not been created yet")

    def americanize(self, european_prices: np.ndarray) -> np.ndarray:
        """
        Exercise the option to receive the payoff immediately if the current value of the option is lower than the payoff if exercised immediately

        Parameters
        ----------
        european_prices : numpy.ndarray
            The European prices (no early exercise)

        Returns
        -------
        numpy.ndarray
            The American prices (after possible early exercise)
        """

        replacement = self.american(self.x_range)
        result = european_prices
        result[result < replacement] = replacement[result < replacement]

        return result

    def explicit(self, t, init_values: np.ndarray,
                 prev_values: np.ndarray) -> np.ndarray:
        """
        A backwards time centered space scheme. As the pde is backwards time, this is an explicit scheme.

        This scheme is:
        * Explicit
        * Conditionally stable (Requires dt >= C (dx)^2 for some C)

        Parameters
        ----------
        t : float
            The current t-value
        init_values : numpy.ndarray
            The t-th PDE solution values
        prev_values : numpy.ndarray
            The (t+dt)-th PDE solution values

        Returns
        -------
        numpy.ndarray
            The t-th PDE solution values
        """
        result = init_values

        # Populate with BTCS scheme
        alpha = self.dt * (- self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) + self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)
        beta = 1 + self.dt * (- 2 * self.diffusion(self.x_range[1:-1], t) / (
            self.dx)**2 + self.source(self.x_range[1:-1], t))
        gamma = self.dt * (self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) + self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)

        result[1:-1] = alpha * prev_values[:-2] + beta * \
            prev_values[1:-1] + gamma * prev_values[2:]

        # If American option, replace with payoff if early exercise is optimal
        if self.american is not None:
            result = self.americanize(result)

        return result

    def implicit(self, t, init_values: np.ndarray,
                 prev_values: np.ndarray) -> np.ndarray:
        """
        A forwards time centered space scheme.

        This scheme is:
        * Implicit
        * Unconditionally stable
        * Oscillations may occur with discontinuities (On the payoff or from more than 1 discrete exercise time)

        Parameters
        ----------
        t : float
            The current t-value
        init_values : numpy.ndarray
            The t-th PDE solution values
        prev_values : numpy.ndarray
            The (t+dt)-th PDE solution values

        Returns
        -------
        numpy.ndarray
            The t-th PDE solution values
        """
        result = init_values

        # Create tridiagonal matrix for FTCS scheme
        alpha = self.dt * (self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) - self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)
        beta = 1 + self.dt * (2 * self.diffusion(self.x_range[1:-1], t) / (
            self.dx)**2 - self.source(self.x_range[1:-1], t))
        gamma = self.dt * (- self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) - self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)

        diagonals = [alpha[1:], beta, gamma[:-1]]

        matrix_step = sparse.diags(diagonals, [-1, 0, 1], dtype=float)

        boundary_conditions = np.zeros(result.shape[0] - 2)
        boundary_conditions[0] = alpha[0] * init_values[0]
        boundary_conditions[-1] = gamma[-1] * init_values[-1]

        inverted_step = sparse.linalg.inv(matrix_step)

        # print(prev_values[-1:1], boundary_conditions)
        result[1:-1] = np.array(inverted_step *
                                (prev_values[1:-1] - boundary_conditions))

        # If American option, replace with payoff if early exercise is optimal
        if self.american is not None:
            result = self.americanize(result)

        return result

    def crank_nicolson(self, t, init_values: np.ndarray,
                       prev_values: np.ndarray) -> np.ndarray:
        """
        The Crank-Nicolson method, the average of the implicit and explicit methods.

        This scheme is:
        * Mixed
        * Unconditionally stable
        * Requires a set of simultaneous equations to be solved on each step
        * Oscillations may occur with discontinuities (From more than 1 discrete exercise time / Bermudan option)

        Parameters
        ----------
        t : float
            The current t-value
        init_values : numpy.ndarray
            The t-th PDE solution values
        prev_values : numpy.ndarray
            The (t+dt)-th PDE solution values

        Returns
        -------
        numpy.ndarray
            The t-th PDE solution values
        """
        result = init_values

        # Create tridiagonal matrix for FTCS scheme
        alpha_explicit = self.dt * (- self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) + self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)
        beta_explicit = 1 + self.dt * \
            (- 2 * self.diffusion(self.x_range[1:-1], t) /
             (self.dx)**2 + self.source(self.x_range[1:-1], t))
        gamma_explicit = self.dt * (self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) + self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)

        diagonals_explicit = [alpha_explicit[1:],
                              beta_explicit, gamma_explicit[:-1]]
        matrix_explicit = sparse.diags(
            diagonals_explicit, [-1, 0, 1], dtype=float)

        # Create diagonals for BTCS scheme
        alpha_implicit = self.dt * (self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) - self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)
        beta_implicit = 1 + self.dt * \
            (2 * self.diffusion(self.x_range[1:-1], t) /
             (self.dx)**2 - self.source(self.x_range[1:-1], t))
        gamma_implicit = self.dt * (- self.convection(self.x_range[1:-1], t) / (
            2 * self.dx) - self.diffusion(self.x_range[1:-1], t) / (self.dx)**2)

        diagonals_implicit = [alpha_implicit[1:],
                              beta_implicit, gamma_implicit[:-1]]
        matrix_implicit = sparse.diags(
            diagonals_implicit, [-1, 0, 1], dtype=float)
        inverted_implicit = sparse.linalg.inv(matrix_implicit)

        # Account for boundary conditions
        boundary_conditions = np.zeros(result.shape[0] - 2)
        boundary_conditions[0] = alpha_explicit[0] * \
            prev_values[0] - alpha_implicit[0] * init_values[0]
        boundary_conditions[-1] = gamma_explicit[-1] * \
            prev_values[-1] - gamma_implicit[-1] * init_values[-1]

        # Populate result
        result[1:-1] = np.array(inverted_implicit * (matrix_explicit *
                                prev_values[1:-1] + boundary_conditions))

        # If American option, replace with payoff if early exercise is optimal
        if self.american is not None:
            result = self.americanize(result)

        return result


class BlackScholesFDM:
    """
    A class containing the various parameters for the Black-Scholes PDE

    Attributes
    ----------
    option : Option
        The option information (K, r, q, etc.)
    N_S : int
        The number of points on the spot-axis
    N_t : int
        The number of points on the time-axis
    S_current : float
        The current spot price
    scheme : CDScheme
        The scheme to use
    """

    def __init__(
            self,
            option: Option,
            N_S,
            N_t,
            S_current,
            scheme=CDScheme.EXPLICIT) -> None:
        """
        A class containing the various parameters for the Black-Scholes PDE

        Parameters
        ----------
        option : Option
            The option information (K, r, q, etc.)
        N_S : int
            The number of points on the spot-axis
        N_t : int
            The number of points on the time-axis
        S_current : float
            The current spot price
        scheme : CDScheme
            The scheme to use
        """
        self.option = option

        # Set grid maximum and minimum to 5 standard deviations away from the
        # current spot.
        half_xrange = 5 * vol * np.sqrt(option.T)

        self.S_max = S_current * np.exp(half_xrange)
        self.S_min = S_current * np.exp(-half_xrange)

        # Throw error if K is out of bounds
        if option.K > self.S_max or option.K < self.S_min:
            raise ValueError(
                f"K is required to be within 5 standard deviations of S_current; {S_current=}, {self.option.K=}")

        # Define upper and lower boundary conditions
        S_boundary_upper = partial(
            self.option.payoff.S_boundary_upper,
            self.S_max,
            self.option.r,
            self.option.T)
        S_boundary_lower = partial(
            self.option.payoff.S_boundary_lower,
            self.S_min,
            self.option.r,
            self.option.T)

        # Ensure there are the same number of intervals to the left and right
        # of S_current
        N_S_odd = N_S + ((N_S + 1) % 2)

        self.pricer = ConvectionDiffusionPricer(
            convection=self.convection,
            diffusion=self.diffusion,
            source=self.source,
            boundary_terminal=self.terminal,
            boundary_xmin=S_boundary_lower,
            boundary_xmax=S_boundary_upper,
            N_x=N_S_odd,
            N_t=N_t,
            x_max=self.S_max,
            x_min=self.S_min,
            t_max=self.option.T,
            american=self.option.payoff if self.option.american else None)

        # Immediately populate grid
        self.pricer.populate_grid(scheme)

    def get_grid(self):
        """
        Returns solution grid from self.pricer

        Returns
        -------
        numpy.ndarray
            The BS PDE solution grid
        """
        return self.pricer.get_grid()

    def get_spot_range(self):
        """
        Returns the spot-axis from self.pricer

        Returns
        -------
        numpy.ndarray
            The S-axis
        """
        return self.pricer.x_range

    def get_time_range(self):
        """
        Returns the time-axis from self.pricer

        Returns
        -------
        numpy.ndarray
            The t-axis
        """
        return self.pricer.t_range

    def convection(self, S, t) -> float:
        """
        The convection term (a = -rS) of the BS PDE

        Parameters
        ----------
        S : float
            The spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The convection coefficient for the given arguments
        """
        return -(self.option.r - self.option.q) * S

    def diffusion(self, S, t) -> float:
        """
        The diffusion term (a = -0.5 * vol^2 * S^2) of the BS PDE

        Parameters
        ----------
        S : float
            The spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The diffusion coefficient for the given arguments
        """
        return -0.5 * (self.option.vol * S)**2

    def source(self, S, t) -> float:
        """
        The source term (c = r) of the BS PDE

        Parameters
        ----------
        S : float
            The spot price
        t : float
            The current time (up to self.option.T)

        Returns
        -------
        float
            The source coefficient for the given arguments
        """
        return (self.option.r)

    def terminal(self, S) -> float:
        """
        The terminal boundary conditions (t=T)

        Parameters
        ----------
        S : float
            The spot price

        Returns
        -------
        float
            The price of the option at the boundary
        """
        return self.option.payoff.get_payoff(S)


if __name__ == "__main__":
    """
    Minimal working example
    """

    from deriv_quant_package.pricer.bsm import BSM
    from deriv_quant_package.volSurface.volsmile_flat import VolSmileFlat
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    # Set up parameters
    K = 0.5
    S_current = 0.5
    r = 0.1
    q = 0.05
    vol = 0.5
    T = 1
    spot_iterations = 75
    call = False
    equal = True
    american = True

    # Initialize option pricing objects
    option = Option(K=K, r=r, q=q, vol=vol, T=T, american=american,
                    payoffclass=DigitalPayoff, call=call, equal=equal)
    BSFDM = BlackScholesFDM(
        option=option,
        N_S=spot_iterations,
        N_t=200,
        S_current=S_current,
        scheme=CDScheme.IMPLICIT)
    BS_pricer = BSM(r=r, T=T, q=q, put=not call)
    flat_vol = VolSmileFlat(vol=vol)

    # Retrieve grid and coordinates
    grid = BSFDM.get_grid()
    S_range = BSFDM.get_spot_range()
    t_range = BSFDM.get_time_range()

    # Retrieve actual prices
    actual_prices = BS_pricer.get_price(
        S=S_range, K=K, vol=flat_vol.get_vol(K))

    # Plot t-slices
    plt.close()
    for i in range(t_range.shape[0]):
        # if i % 10 == 0:
        plt.plot(S_range, grid[:, i])
    plt.title = (
        f"{'American' if american else 'European'} {'call' if call else 'put'} option prices")
    plt.xlabel("S")
    plt.ylabel("V")
    plt.draw()

    # Plot 3d surface of option prices
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.contourf(S_range, t_range, grid.T, 200, cmap="jet")
    ax.set_title("")
    ax.set_xlabel('S')
    ax.set_ylabel('t')
    ax.set_zlabel('V')
    plt.draw()
    plt.show()


def thomas(mat: np.ndarray, x: np.ndarray, d: np.ndarray):
    """
    Solves a tridiagonal system in O(n) time
    Known to be stable if matrix is diagonally dominant or symmetric postiive definite
    TODO: to replace scipy.linalg.solve with this
    """
    dimX = mat.shape[0]
    dimY = mat.shape[1]

    assert dimX == dimY, "mat must be a square matrix"

    a = np.diagonal(mat, offset=-1).copy()
    b = np.diagonal(mat).copy()
    c = np.diagonal(mat, offset=1).copy()

    # Forward sweep
    c_prime = np.zeros(c.shape[0])
    d_prime = np.zeros(dimX)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, c_prime.shape[0]):
        c_prime[i] = c[i] / (b[i] - a[i - 1] * c_prime[i - 1])

    for i in range(1, d_prime.shape[0]):
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / \
            (b[i] - a[i - 1] * c_prime[i - 1])

    result = np.zeros(dimX)
    result[-1] = d_prime[-1]
    for i in range(dimX - 2, -1, -1):
        result[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return result
