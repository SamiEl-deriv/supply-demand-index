from ..fdm import FiniteDifference
from ...utils import tridiagonal_solver
import numpy as np
import scipy.sparse as sparse


class FDMEulerCrankNicolson(FiniteDifference):

    def step_march(self, t, init_values: np.ndarray,
                   prev_values: np.ndarray) -> np.ndarray:
        """
        The Crank-Nicolson method, the average of the implicit and explicit methods.

        This scheme is:
        * Mixed
        * Unconditionally stable (in L2-norm);
        * Spurious oscillations may arise if the function is not in L2
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

        a_i = self._dt * \
            self._PDE.coefficient_convection(
                self._Xrange[1:-1], t + self._dt) / (2 * self._dx)
        b_i = self._dt * \
            self._PDE.coefficient_diffusion(
                self._Xrange[1:-1], t + self._dt) / (self._dx)**2
        c_i = self._dt * \
            self._PDE.coefficient_zero(self._Xrange[1:-1], t + self._dt)

        a_e = self._dt * \
            self._PDE.coefficient_convection(
                self._Xrange[1:-1], t) / (2 * self._dx)
        b_e = self._dt * \
            self._PDE.coefficient_diffusion(
                self._Xrange[1:-1], t) / (self._dx)**2
        c_e = self._dt * self._PDE.coefficient_zero(self._Xrange[1:-1], t)

        # Create tridiagonal matrix for FTCS scheme
        alpha_explicit = (-a_e + b_e) / 2
        beta_explicit = 1 - b_e + c_e / 2
        gamma_explicit = (a_e + b_e) / 2

        diagonals_explicit = [alpha_explicit[1:],
                              beta_explicit, gamma_explicit[:-1]]
        matrix_explicit = sparse.diags(
            diagonals_explicit, [-1, 0, 1], dtype=float)

        # Create diagonals for BTCS scheme
        alpha_implicit = (a_i - b_i) / 2
        beta_implicit = 1 + b_i - c_i / 2
        gamma_implicit = - (a_i + b_i) / 2

        diagonals_implicit = [alpha_implicit[1:],
                              beta_implicit, gamma_implicit[:-1]]
        matrix_implicit = sparse.diags(
            diagonals_implicit, [-1, 0, 1], dtype=float)

        # Account for boundary conditions
        boundary_conditions = np.zeros(result.shape[0] - 2)
        boundary_conditions[0] = alpha_explicit[0] * \
            prev_values[0] - alpha_implicit[0] * init_values[0]
        boundary_conditions[-1] = gamma_explicit[-1] * \
            prev_values[-1] - gamma_implicit[-1] * init_values[-1]
        # Calculate source terms
        source_terms = self._dt / 2 * (self._PDE.coefficient_source(
            self._Xrange[1:-1], t) + self._PDE.coefficient_source(self._Xrange[1:-1], t + self._dt))

        # Populate result

        # Choose tridiagonal solver if selected, otherwise use standard matrix
        # inversion methods
        if self._mode == "fast":
            result[1:-1] = np.array(tridiagonal_solver(
                mat=matrix_implicit.A,
                d=matrix_explicit * prev_values[1:-1] + boundary_conditions + source_terms))
        elif self._mode == "gaussian":
            inverted_implicit = sparse.linalg.inv(matrix_implicit)
            result[1:-
                   1] = np.array(inverted_implicit *
                                 (matrix_explicit *
                                  prev_values[1:-
                                              1]) +
                                 boundary_conditions +
                                 source_terms)
        else:
            raise ValueError("Invalid matrix inversion mode")

        # If American option, replace with payoff if early exercise is optimal
        if self._American is not None:
            result = self.americanize(result)

        return result
