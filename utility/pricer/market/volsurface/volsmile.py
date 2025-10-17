from abc import ABC, abstractmethod


class VolSmile(ABC):
    """
    An abstract volatility surface class
    """

    def __call__(self, *args) -> float:
        """
        Wrapper for get_vol

        Returns
        -------
        float
            Flat volatility
        """

        return self.get_vol(args)

    @abstractmethod
    def __str__(self) -> str:
        """
        Prints volatility smile details
        """

        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def get_vol(self, *args) -> float:
        """
        Abstract function for returning volatility
        """

        raise NotImplementedError("Not Implemented")
