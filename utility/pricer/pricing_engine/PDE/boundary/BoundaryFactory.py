from .Boundary import Boundary
from .Boundaries import *


boundariesRefDict : dict[str, Type[Boundary]] = {
    "call"            : BoundaryCall          ,
    "put"             : BoundaryPut           ,
    "up_out_call"     : BoundaryUpOutCall     ,
    "up_in_call"      : BoundaryUpInCall      ,
    "down_out_call"   : BoundaryDownOutCall   ,
    "down_in_call"    : BoundaryDownInCall    ,
    "down_out_put"    : BoundaryDownOutPut    ,
    "down_in_put"     : BoundaryDownInPut     ,
    "up_out_put"      : BoundaryUpOutPut      ,
    "up_in_put"       : BoundaryUpInPut       ,
    "digital_call"    : BoundaryDigitalCall   ,
    "digital_put"     : BoundaryDigitalPut    ,
    "sharkfinKO_call" : BoundarySharkfinKOCall,
    "sharkfinKO_put"  : BoundarySharkfinKOPut ,
    "sharkfinXP_call" : BoundarySharkfinXPCall,
    "sharkfinXP_put"  : BoundarySharkfinXPPut ,
}


class BoundaryFactory:
    """
    BoundaryFactory class
    """
    __builders = boundariesRefDict

    @classmethod
    def CreateBoundary(cls, PayOffId: str, **kwargs) -> Boundary:
        """
        Creates new boundary conditions using a stored builder

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
            builder = cls.__builders[PayOffId]
        except KeyError:
            raise ValueError("Selected PayOff boundary not supported\n"
                             f"Try one of {list(cls.__builders.keys())}")
        return builder(**kwargs)
