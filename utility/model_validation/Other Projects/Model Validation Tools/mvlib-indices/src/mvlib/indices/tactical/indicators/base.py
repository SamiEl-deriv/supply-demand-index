from abc import ABC, abstractmethod

class Signal(ABC):
    '''
    Base Signal Class.
    '''
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("Not yet implemented!")
    
    def __call__(self, *args, **kwds):
        raise NotImplementedError("Not yet implemented!")
    
class Strategy(ABC):
    """
    Base Strategy class.
    """
    def __init__(self) -> None:
        return NotImplementedError("Requires Implementation!")
    
    @abstractmethod
    def get_signal(self):
        return NotImplementedError("Requires Implementation!")