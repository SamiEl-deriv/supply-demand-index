#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:14:55 2023

@author: benoitdalmeida
"""

from abc import ABC, abstractmethod


# =============================================================================
# =============================================================================

class PayOff(ABC):

    @abstractmethod
    def value(self):
        raise NotImplementedError('Implementation required!')


class VanillaOption():
    def __init__(self, PayOff, Expiry, parameters):
        self.PayOff = PayOff  # copy object to be checked
    # .......... to be extended eventually ..............


class PayOffCall(PayOff):

    def __init__(self, strike_):
        self.strike = strike_

    def value(Spot):
        return max(spot - self.strike, 0)

    # .......... to be extended eventually ..............


class PayOffPut(PayOff):

    def __init__(self, Strike_):
        self.Strike = Strike_

    def value(Spot):
        return max(0, self.Strike - Spot)


# In object_factory.py

class PayOffFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, PayOffId, builder):
        self._builders[PayOffId] = builder

    def createPayoff(self, PayOffId, **kwargs):
        builder = self._builders.get(PayOffId)
        if not builder:
            raise ValueError(PayOffId)
        return builder(**kwargs)

    # .......... to be extended eventually ..............


'''
PayOffBridge would have been useful in C++ to manage the memory but won't be of any help in python'

class payOffBridge :
      #is like the memory manager of payoff
      #parameter is only a pointer on a payoff object
      #Method :bridge pattern and rule of three;ensure assigmnent-construction-destruction of the payoff

Generic class enable to create object via template

class helperPayoff:
      #is a template class that as no attribute
      #no attribute just an instance just only create registration using the register method of class payoff factory
      #and by specifying the template class argument T the instance will automatically create the payoff of type T by calling the class T construnctor


'''

# --------------------------------------------------------------------------------------------------------
# convection diffusion equation - second order PDE


class PDEconvectionDiffusion(ABC):
    # attribute
    def __init__(self,):
        pass
    # pde coefficients

    @abstractmethod
    def coefficient_convection(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def coefficient_diffusion(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def coefficient_source(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def zero_coefficient(self):
        raise NotImplementedError('Implementation required!')

   # boundary and init condition
    @abstractmethod
    def boundary_left(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def boundary_right(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def init_cond(self):
        raise NotImplementedError('Implementation required!')


class PDEBlackScholes(PDEconvectionDiffusion):

    def __init__(self, option):
        pass

    def coefficient_convection(self, x, t):
        return

    def coefficient_diffusion(self, x, t):
        return

    def coefficient_source(self):
        return

    def zero_coefficient(self):
        return

    def boundary_left(self):
        return

    def boundary_right(self):
        return

    def init_cond(self):
        return


class FiniteDifference(ABC):

    def __init__(self, _x_dom, _J, _t_dom, _N, _pde):
        # space discredization
        # spatial extent
        self.x_dom = _x_dom

        # number of special differencing points
        self.J = _J
        # temporal step size
        self.dx
        # time discredisation

        # temporal extent
        self.t_dom = _t_dom

        # coordinates of the x dimension
        self.xvalue

        # Number of temporal differencing points
        self.N = _N

        # temporal step size (to be calculated)
        self.dt

        self.pde = _pde

        # time marching
        self.current_time
        self.previous_time

        # differencing coefficient
        self.alpha
        self.beta
        self.gamma

        # storage
        # New result
        self.new_result
        # oldresult
        self.old_result

    @abstractmethod
    def calculate_step_size(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def set_initial_condition(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def calculate_boundary_condition(self):
        raise NotImplementedError('Implementation required')

    @abstractmethod
    def interpolate(self):
        raise NotImplementedError('Implementation required')

    @abstractmethod
    def calculate_inner_domain():
        raise NotImplementedError('Implementation required')

    @abstractmethod
    def step_march():
        raise NotImplementedError('Implementation required')


class FDMEulerExplicit(FiniteDifference):

    def calculate_step_sizes():
        pass

    def set_initial_condition():
        pass

    def calculate_boundary_conditions():
        pass

    def calculate_inner_domain():
        pass

    def step_march():
        pass
