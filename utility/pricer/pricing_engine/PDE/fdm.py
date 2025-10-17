from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from .pde_params.pdebs import PDEConvectionDiffusion
from ..utils import bilinear_interpolate


class FiniteDifference(ABC):
    """
    A class that solves a backwards-time convection-diffusion equation given the pde:
     V_t = a * V_x + b * V_{xx} + cV + Q

    We call the coefficients:

    a : convection term
    b : diffusion term
    c : source term

    To obtain the grid of solutions, use get_grid.

    There is consideration for American options, set american to None if you want a typical PDE solver
    """

    def __init__(
            self,
            PDE: PDEConvectionDiffusion,
            N_x: int,
            N_t: int,
            mode: str,
            American=False) -> None:
        """
        Parameters
        ----------
        PDE : PDEConvectionDiffusion
            An instance of the PDEConvectionDiffusion, containing all parameters and coefficients
        N_x : int
            The number of points on the x-axis
        N_t : int
            The number of points on the t-axis
        American : boolean
            If True, the PDE solver will replace the current value with the payoff if the latter is higher.
            If False, the PDE solver works as usual
                mode : str ("fast" | "gaussian")
            "fast":
            The tridiagonal matrix solver (Thomas algorithm) will be used.
                Complexity: O(n)
                Pros: Faster than "gaussian"
                Cons: Does not work with all tridiagonal matrices (requires at least positive semi-definite or diagonally dominant matrices)
            "gaussian":
                Gaussian matrix inversion (numpy.linalg.inv) will be used.
                Complexity: O(n^3)
                Pros: Works with any invertible matrix
                Cons: Much slower than "fast" for large n
        """

        self._PDE = PDE
        self._mode = mode

        # Set parameters for uniform grid
        self._Nx = N_x
        self._Nt = N_t
        self._Xmin = PDE.smin
        self._Xmax = PDE.smax
        self._Tmax = PDE.option.expiry

        # Initialize grid
        self._Grid = None
        self.setup_grid()
        # apply a shift of the grid if the critical points tuple is not empty
        critical_point = PDE.option.payOff.critical_point
        if len(critical_point) != 0:
            self.shift_Xgrid(critical_point)

        # American replacement condition
        self._American = PDE.option.payOff if American else None

        # Populate grid
        self.calculate_inner_domain()

    def setup_grid(self) -> None:
        """
        Initializes uniform spacing, grids and coordinate arrays. Coordinates are in terms of (x,t) and confined in the rectangle [x_min, x_max] x [0, t_max]
        TODO: To modify once grid class is created

        Parameters
        ----------
        _dx : float
            distance between points in x-array
        _dt : float
            distance between points in t-array
        _Xrange : np.ndarray
            Discrete x-axis points
        _Trange : np.ndarray
            Discrete t-axis points
        _Grid : np.ndarray
            Discrete grid with coordinates (x, t) in [0, x_range] x [0, t_range]
        """

        # There will be N_x-1 space intervals
        self._dx = (self._Xmax - self._Xmin) / (self._Nx - 1)

        # Ensure there are N_t - 1 intervals
        self._dt = - self._Tmax / (self._Nt - 1)

        # Initialize uniform ranges
        self._Xrange = np.linspace(
            self._Xmin, self._Xmax, num=self._Nx, endpoint=True)
        self._Trange = np.linspace(0, self._Tmax, num=self._Nt, endpoint=True)

        # Initialize grid
        self._Grid = np.zeros((self._Nx, self._Nt))

    def shift_Xgrid(self, critical_point: tuple):
        """
        Parameters
        ----------
        critical_point: tuple
             A tuple containing the point of discontinuity of the payoff or greek

        One or more critical points have a specific position on the grid (e.g,a point should lie exactly on grid a point,
        or midway betweeb two grid points) but that the grid may be otherwise uniform. That is, we construct a grid that
        is nearly uniform and smoothly varying,but 'pinned' at the critical points.In such case, spline transformations
        are particulary useful.
        The critical point approximately (but not exactly) in the middle of two grid points [Tavella and Randall 2000, p. 171].
        The advantage of the cubic spline smooth deformation is to preserve the second-order convergence.
        Let {B_k} be the set of critical points where 1<=k <= K  and assume 0<=epsilon<=1.Then we may find a set of
        associated grid points {epsilon_k},epsilon_k = 1/I round((B_k-S_min)I/(S_max-S_min))
        """
        # Convert tuple to array
        array_critical_point = np.array(critical_point)

        # Adding the critical point to the spatial range of the intial X grid
        X_grid_with_critical = np.sort(np.concatenate(
            (self._Xrange, array_critical_point)), kind='mergesort')
        _epsilon_range = (1 / self._Nx) * np.round_((self._Xrange -
                                                     self._Xmin) * self._Nx / (self._Xmax - self._Xmin))

        # Preparing the shift of all points on the grid included the critical
        # points
        _epsilon_critical = (1 / self._Nx) * np.round_((X_grid_with_critical - self._Xmin)
                                                       * self._Nx / (self._Xmax - self._Xmin)) + 1 / (2 * self._Nx)
        _epsilon_critical = _epsilon_critical[_epsilon_critical <= 1]
        # Fitting the cubic spline on initial X grid
        cs = CubicSpline(_epsilon_range, self._Xrange)

        # Shifting of all points on the initial grid included the added
        # critical points to generate a new grid
        X_grid_shift = np.unique(cs(_epsilon_critical))

        # Updating the X grid; the old points in the X grid are for large Nx
        # close to the middle point of the X_grid_shift
        self._Xrange = X_grid_shift

        """IMPORTANT: Adjust grid, Xmin/max, Nx and boundaries to prevent broadcasting & numerical errors"""
        # Adjust Xmin & Xmax
        self._Xmax = self._Xrange[-1]
        self._Xmin = self._Xrange[0]

        # Adjust Nx
        self._Nx = self._Xrange.shape[0]

        # Adjust grid
        self._Grid = np.zeros((self._Nx, self._Nt))

        # Update boundary Smin, Smax so that the boundary conditions will
        # return corretly
        self._PDE.boundary.set_spot_upper(self._Xmax)
        self._PDE.boundary.set_spot_lower(self._Xmin)

    def set_initial_condition(self):
        """
        Sets initial (terminal) conditions of grid
        """
        self._Grid[:, -1] = self._PDE.option.payOff(self._Xrange)

    def calculate_boundary_conditions(self):
        # # Assign min/max conditions
        self._Grid[0, :-
                   1] = self._PDE.boundary_spot_lower(self._Trange[:-
                                                                   1]) if not self._American else self._American(self._Xmin)
        self._Grid[-1, :-1] = self._PDE.boundary_spot_upper(
            self._Trange[:-1]) if not self._American else self._American(self._Xmax)

    def calculate_inner_domain(self) -> None:
        """
        Populates grid with solution of the pde
        """

        # Set up grid and boundary conditions
        self.set_initial_condition()
        self.calculate_boundary_conditions()

        # Iterate backwards and populate grid
        for i in range(self._Trange.shape[0] - 2, -1, -1):
            self._Grid[:, i] = self.step_march(
                t=self._Trange[i], init_values=self._Grid[:, i], prev_values=self._Grid[:, i + 1])

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

        result = european_prices

        # Replace if payoff exceeds current value
        replacement = self._American(self._Xrange)
        result[result < replacement] = replacement[result < replacement]

        return result

    def get_x_range(self) -> np.ndarray:
        """
        Retrieves the space-lattice points

        Returns
        -------
        numpy.ndarray
            The x-ordinates of the grid
        """
        return self._Xrange

    def get_t_range(self) -> np.ndarray:
        """
        Retrieves the time-lattice points

        Returns
        -------
        numpy.ndarray
            The t-ordinates of the grid
        """
        return self._Trange

    def get_surface(self, with_coords=False) -> np.ndarray:
        """
        Retrieves a populated grid - solution to the pde.

        * For specific values, use get_value
        * For quick access to solutions at t=0, use get_solution_range

        Examples
        --------

        # without coords
        grid = FDM.get_solution()

        # with coords
        coord_grid = FDM.get_solution(with_coords = True)

        # Slicing with coords
        x_slice  = coord_grid[i,:,:]
        t_slice  = coord_grid[:,j,:]

        # Getting ranges from grid
        x_range  = coord_grid[:,:,0]
        t_range  = coord_grid[:,:,1]
        z_values = coord_grid[:,:,2]

        Parameters
        ----------
        with_coords : bool (DEFAULT = False)
            If true, return 2d numpy array containing coordinates and solution -- elements of form [x,t,f(x,t)]
            Else return 2d numpy array of f(x,t)

        Returns
        -------
        numpy.ndarray
            Returns grid with or without coordinates
        """

        grid = self._Grid

        # Return error if grid has been populated
        if grid is None:
            return ValueError("Grid has not been created yet")

        # Return grid only
        if not with_coords:
            return grid

        # Set up meshes for grid coordinates
        x_range = self.get_x_range()
        t_range = self.get_t_range()
        xx, tt = np.meshgrid(x_range, t_range, indexing="ij")

        # Cast to array of [x,t,f(x,t)]'s
        # Transpose is necessary to keep x's and t's in proper indices
        return np.dstack((xx, tt, grid))

    def get_solution_range(self, with_coords: bool = False) -> np.ndarray:
        """
        Retrieves the solution to the PDE at t=0

        Examples
        --------

        graph = FDM.get_solution_range()
        x_range = graph[:,0]
        solutions = graph[:,1]

        For particular x-coordinate solution, use get_value

        Parameters
        ----------
        with_coords : bool (DEFAULT = False)
            If true, return numpy array containing x-ordinates and solution -- elements of form [x, f(x,0)]
            Else return numpy array of f(x,0)

        Returns
        -------
        numpy.ndarray
            The solution at time t=0 with or without coordinates
        """

        # Return error if grid has been populated
        if self._Grid is None:
            return ValueError("Grid has not been created yet")

        solution = self._Grid

        if not with_coords:
            return solution

        x_range = self.get_x_range()

        # Cast to to array of [x,f(x,0)]'s
        return np.dstack((x_range, self._Grid[:, 0]))[0]

    def get_value(self, x: float, t: float) -> float:
        """
        Retrieves the price given (x,t) coordinates using bilinear interpolation

        Parameters
        ----------
        x : float
            The x-coordinate to search for
        t : float
            The t-coordinate to search for

        Returns
        -------
        float
            The interpolated z-value
        """

        x_range = self.get_x_range()
        t_range = self.get_t_range()

        # Check if given values are out of bounds
        if x < x_range[0] or x > x_range[-1]:
            raise ValueError(
                f"x-value is out of range: x_min = {x_range[0]}; x_max = {x_range[-1]}")
        if t < t_range[0] or t > t_range[-1]:
            raise ValueError(
                f"t-value is out of range: t_min = {t_range[0]}; t_max = {t_range[-1]}")

        z_surface = self.get_surface()

        # Get nearest left-neighbours
        x_left = np.searchsorted(x_range, x) - 1
        t_left = np.searchsorted(t_range, t) - 1

        square = np.array([[x_range[i], t_range[j], z_surface[i, j]] for i in [
                          x_left, x_left + 1] for j in [t_left, t_left + 1]])

        return bilinear_interpolate(square, x, t)

    @abstractmethod
    def step_march(self, t, init_values: np.ndarray,
                   prev_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Implementation required')
