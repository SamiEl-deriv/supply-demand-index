from .options import Option
from .options import *

from typing import Type

optionsRefDict : dict[str, Type[Option]] = {
    "call"            : VanillaOption,
    "put"             : VanillaOption,
    "up_out_call"     : VanillaOption,
    "up_in_call"      : BarrierOption,
    "down_out_call"   : BarrierOption,
    "down_in_call"    : BarrierOption,
    "down_out_put"    : BarrierOption,
    "down_in_put"     : BarrierOption,
    "up_out_put"      : BarrierOption,
    "up_in_put"       : BarrierOption,
    "digital_call"    : DigitalOption,
    "digital_put"     : DigitalOption,
    "sharkfinKO_call" : SharkfinOption,
    "sharkfinKO_put"  : SharkfinOption,
    "sharkfinXP_call" : SharkfinOption,
    "sharkfinXP_put"  : SharkfinOption
}


class OptionFactory:
    """
    OptionFactory class
    """
    _builders = optionsRefDict

    @classmethod
    def CreateOption(cls, OptionId: str, **kwargs):
        """
        Creates a new option contract using a stored builder

        Parameters
        ----------
            OptionID : str
                The option ID for the builder
            **kwargs
                arguments of the option contract constructor -> float

        Returns
        -------
        Option
            Option class instance

        """
        try:
            builder = cls._builders[OptionId]
        except KeyError:
            raise ValueError("Selected option not supported\n"
                             f"Try one of {list(cls._builders.keys())}")
        return builder(**kwargs)
