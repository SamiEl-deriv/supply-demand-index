from .payoff import *

from typing import Type

payOffsRefDict : dict[str, Type[PayOff]] = {
    "call"            : PayOffCall        ,
    "put"             : PayOffPut         ,
    "up_out_call"     : PayOffUpOutCall   ,
    "up_in_call"      : PayOffUpInCall    ,
    "down_out_call"   : PayOffDownOutCall ,
    "down_in_call"    : PayOffDownInCall  ,
    "down_out_put"    : PayOffDownOutPut  ,
    "down_in_put"     : PayOffDownInPut   ,
    "up_out_put"      : PayOffUpOutPut    ,
    "up_in_put"       : PayOffUpInPut     ,
    "digital_call"    : PayOffDigitalCall ,
    "digital_put"     : PayOffDigitalPut  ,
    "sharkfinKO_call" : PayOffSharkfinCall,
    "sharkfinKO_put"  : PayOffSharkfinPut ,
    "sharkfinXP_call" : PayOffSharkfinCall,
    "sharkfinXP_put"  : PayOffSharkfinPut
}


class PayOffFactory:
    """
    PayOffFactory class
    """
    _builders = payOffsRefDict

    @classmethod
    def CreatePayoff(cls, PayOffId: str, **kwargs):
        """
        Creates a new payoff using a stored builder

        Parameters
        ----------
            BoundaryID : str
                The boundary ID for the builder
            **kwargs
                arguments of the boundary constructor -> float

        Returns
        -------
        Boundary
            Boundary class instance

        """
        try:
            builder = cls._builders[PayOffId]
        except KeyError:
            raise ValueError("Selected PayOff not supported\n"
                             f"Try one of {list(cls._builders.keys())}")
        return builder(**kwargs)
