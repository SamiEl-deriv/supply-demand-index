from abc import ABC, abstractmethod
from ....options.options import Option


class PDEConvectionDiffusion(ABC):
    """
    A class that contains the parameters for a backwards-time semi-linear convection-diffusion equation:
     V_t = a * V_x + b * V_{xx} + cV + Q

    We call the coefficients:

    a(x,t) : convection coefficient
    b(x,t) : diffusion coefficient
    c(x,t) : zero coefficient
    Q(x,t) : source function

    """
    # attribute

    def __init__(self, Option: Option):
        self.__Option = Option
        self._Smax = None
        self._Smin = None
        self._Boundary = None
        self._FDM = None

    def get_solution_range(self):
        return self._Pricer.get_solution()
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
    def coefficient_zero(self):
        raise NotImplementedError('Implementation required!')

    # boundary and init condition
    @abstractmethod
    def boundary_spot_lower(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def boundary_spot_upper(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def init_cond(self):
        raise NotImplementedError('Implementation required!')

    @property
    def option(self) -> Option:
        return self.__Option

    @property
    def smin(self) -> Option:
        return self._Smin

    @property
    def smax(self) -> Option:
        return self._Smax

    @property
    def boundary(self) -> Option:
        return self._Boundary
