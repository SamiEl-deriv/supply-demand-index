#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..fdm import FiniteDifference
from ...utils import tridiagonal_solver
import numpy as np
import scipy.sparse as sparse


class FDMEulerCrankNicolsonRannacher(FiniteDifference):

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
            Discrete grid with coordinates (x, t) in [x_min, x_max] x [0, t_max]
        """

        # There will be N_x-1 space intervals if N_x is odd or N_x space
        # intervals if N_x is even to ensure that S_current is on a grid point
        self._dx = (self._Xmax - self._Xmin) / (self._Nx - 1)

        # Ensure there are N_t - 1 intervals regardless of parity of N_t
        self._dt = - self._Tmax / (self._Nt - 1)

        # Initialize uniform ranges
        self._Xrange = np.linspace(self._Xmin, self._Xmax, self._Nx)
        self._Trange = np.linspace(0, self._Tmax, self._Nt)

        # add half time step for Rannacher time-marching
        first_half_step = (self._Trange[-2] + self._Trange[-1]) / 2
        second_half_step = (self._Trange[-3] + self._Trange[-2]) / 2
        self._Trange = np.insert(self._Trange, -1, first_half_step)
        self._Trange = np.insert(self._Trange, -3, second_half_step)

        # Reset Nt
        self._Nt = self._Trange.shape[0]
        # Initialize grid
        self._Grid = np.zeros((self._Nx, self._Nt + 2))

    def step_march(self, t, init_values: np.ndarray,
                   prev_values: np.ndarray) -> np.ndarray:
        """
        The Crank-Nicolson Rannacher  method objective was to recover second-order convergence in the context of Crank–Nicolson time-marching
        (he also considered higher-order time integration schemes), and using energy methods he proved that this could be achieved by replacing
        the Crank–Nicolson approximation for the very first timestep by two half-timesteps using backward Euler time integration.
        four half-timesteps of backward Euler time-marching is the minimum required to recover second-order convergence for these two problems.
        The use of more than four half-timesteps will lead to an increase in the overall error, due to the lower-order discretization error inherent
        in the backward Euler discretization, and therefore four half-timesteps can be regarded as optimal.

        Reference:
        Convergence analysis of Crank–Nicolson and Rannacher time-marching
        Michael B. Giles
        Oxford University Computing Laboratory, Wolfson Building, Parks Road, Oxford OX1 3QD, UK
        Rebecca Carter

        * Mixed
        *mixed timestep
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
        # consideration for Rannacher time-marching procedure
        result = init_values
        self.dt_rannacher = self._dt / 2
        if t in self._Trange[-5:]:

            """
            A Backwards time centered space scheme.

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

            a = self.dt_rannacher * \
                self._PDE.coefficient_convection(
                    self._Xrange[1:-1], t) / (2 * self._dx)
            b = self.dt_rannacher * \
                self._PDE.coefficient_diffusion(
                    self._Xrange[1:-1], t) / (self._dx)**2
            c = self.dt_rannacher * \
                self._PDE.coefficient_zero(self._Xrange[1:-1], t)

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

            # Choose tridiagonal solver if selected, otherwise use standard
            # matrix inversion methods
            if self._mode == "fast":
                result[1:-
                       1] = np.array(tridiagonal_solver(mat=matrix_step.A, d=prev_values[1:-
                                                                                         1] -
                                     boundary_conditions +
                                     self.dt_rannacher *
                                     self._PDE.coefficient_source(self._Xrange[1:-
                                                                               1], t +
                                                                  self.dt_rannacher)))
            elif self._mode == "gaussian":
                inverted_step = sparse.linalg.inv(matrix_step)
                result[1:-
                       1] = np.array(inverted_step *
                                     (prev_values[1:-
                                                  1] -
                                      boundary_conditions +
                                      self.dt_rannacher *
                                      self._PDE.coefficient_source(self._Xrange[1:-
                                                                                1], t +
                                                                   self.dt_rannacher)))
            else:
                raise ValueError("Invalid matrix inversion mode")

            # If American option, replace with payoff if early exercise is
            # optimal
            if self._American is not None:
                result = self.americanize(result)

        else:

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

            # Choose tridiagonal solver if selected, otherwise use standard
            # matrix inversion methods
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

            # If American option, replace with payoff if early exercise is
            # optimal
            if self._American is not None:
                result = self.americanize(result)

        return result
