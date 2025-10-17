from ..fdm import FiniteDifference
from ...utils import tridiagonal_solver
import numpy as np
import scipy.sparse as sparse


class FDMEulerImplicit(FiniteDifference):

    def step_march(self, t, init_values: np.ndarray,
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

        a = self._dt * \
            self._PDE.coefficient_convection(
                self._Xrange[1:-1], t) / (2 * self._dx)
        b = self._dt * \
            self._PDE.coefficient_diffusion(
                self._Xrange[1:-1], t) / (self._dx)**2
        c = self._dt * self._PDE.coefficient_zero(self._Xrange[1:-1], t)

        # Create tridiagonal matrix for FTCS scheme
        alpha = a - b
        beta = 1 + 2 * b - c
        gamma = -a - b

        diagonals = [alpha[1:], beta, gamma[:-1]]

        # TODO: Remove sparse matrix construction, unnecessary
        matrix_step = sparse.diags(diagonals, [-1, 0, 1], dtype=float)

        # TODO: Replace with boundary factory usage
        boundary_conditions = np.zeros(result.shape[0] - 2)
        boundary_conditions[0] = alpha[0] * init_values[0]
        boundary_conditions[-1] = gamma[-1] * init_values[-1]

        # Choose tridiagonal solver if selected, otherwise use standard matrix
        # inversion methods
        if self._mode == "fast":
            result[1:-
                   1] = np.array(tridiagonal_solver(mat=matrix_step.A, d=prev_values[1:-
                                                                                     1] -
                                                    boundary_conditions +
                                                    self._dt *
                                                    self._PDE.coefficient_source(self._Xrange[1:-
                                                                                              1], t +
                                 self._dt)))
        elif self._mode == "gaussian":
            inverted_step = sparse.linalg.inv(matrix_step)
            result[1:-
                   1] = np.array(inverted_step *
                                 (prev_values[1:-
                                              1] -
                                  boundary_conditions +
                                  self._dt *
                                  self._PDE.coefficient_source(self._Xrange[1:-
                                                                            1], t +
                                  self._dt)))
        else:
            raise ValueError("Invalid matrix inversion mode")

        # If American option, replace with payoff if early exercise is optimal
        if self._American is not None:
            result = self.americanize(result)

        return result
