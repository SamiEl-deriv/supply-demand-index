from abc import ABC
from ...options.options import Option


class ClosedFormula(ABC):

    def __init__(self, Option: Option, spot: float) -> None:
        """
        Parameters
        ----------
        Option : VanillaOption
            An instance of the VanillaOption,
            containing a PayOff and a expiracy
        """
        self._spot = spot
        self._Option = Option
        self._name = self._Option.payOff.__class__.__name__

    def set_name(self) -> None:
        """
        Return the name of the class of the VanillaOption instance
        """
        self._name = self._Option.payOff.__name__

#    @abstractmethod
#     def interpolate(self):
#         raise NotImplementedError('Implementation required')
