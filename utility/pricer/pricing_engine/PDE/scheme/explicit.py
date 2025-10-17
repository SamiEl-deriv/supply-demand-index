from ..fdm import FiniteDifference
import numpy as np


class FDMEulerExplicit(FiniteDifference):

    def step_march(self, t, init_values: np.ndarray,
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

        a = self._dt * \
            self._PDE.coefficient_convection(
                self._Xrange[1:-1], t) / (2 * self._dx)
        b = self._dt * \
            self._PDE.coefficient_diffusion(
                self._Xrange[1:-1], t) / (self._dx)**2
        c = self._dt * self._PDE.coefficient_zero(self._Xrange[1:-1], t)

        # Populate with BTCS scheme
        alpha = -a + b
        beta = 1 - 2 * b + c
        gamma = a + b

        result[1:-1] = alpha * prev_values[:-2] + beta * prev_values[1:-1] + gamma * \
            prev_values[2:] + self._dt * \
            self._PDE.coefficient_source(self._Xrange[1:-1], t)

        # If American option, replace with payoff if early exercise is optimal
        if self._American is not None:
            result = self.americanize(result)

        return result
