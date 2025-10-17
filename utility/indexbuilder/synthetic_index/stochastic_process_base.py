from abc import ABC, abstractmethod
from typing import Type, Optional
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from spread_model import spread_dict

from time import sleep

dt = 1/(365 * 24 * 3600)  # 1 second


class StochasticProcess(ABC):

    def __init__(self,
                 start_val: float,
                 dt: float = dt,
                 max_q_size: int = 1000, 
                 spread_model: str = 'default',
                 spread_param: dict = {}) -> None:
        self.__dt = dt
        self.__start_val = start_val
        self.__maxlen = max_q_size
        self.__spread_model = spread_model
        self.__spread_param = spread_param
        self.reset()

    def reset(self):
        self.__index: deque[tuple[float, float]] = deque(maxlen=self.__maxlen)
        self.__bid: deque[tuple[float, float]] = deque(maxlen=self.__maxlen)
        self.__ask: deque[tuple[float, float]] = deque(maxlen=self.__maxlen)
        self.__spread: deque[tuple[float, float]] = deque(maxlen=self.__maxlen)
        self.__last_value = None
        self.__make_spread()

    @property
    def dt(self) -> float:
        return self.__dt
    
    @property
    def maxlen(self) -> int:
        return self.__maxlen
    @maxlen.setter
    def maxlen(self, value):
        self.__maxlen = value
        self.reset()
    
    @property
    def start_val(self) -> float:
        return self.__start_val
    
    @property
    def dt(self) -> int:
        return self.__dt
    @dt.setter
    def dt(self, value):
        self.__dt = value
        self.reset()

    @property
    def index(self):
        return self.__index

    @property
    def bid(self):
        return self.__bid

    @property
    def ask(self):
        return self.__ask
    
    @property
    def spread(self):
        return self.__spread    

    def new_value(self, prev_value: float,
                  seed: Optional[int] = None) -> float:
        return prev_value * np.exp(self.new_return())

    @abstractmethod
    def new_return(self, seed: Optional[int] = None) -> float:
        raise NotImplementedError('new_return method not implemented')
    
    def update(self) -> None:
        if not self.__last_value:
            self.__last_value = self.start_val
        self.__index.append(self.new_value(self.__last_value))
        self.__last_value = self.__index[-1]
        self.compute_spread(**self.__spread_param)

    def make_index(self, new_len=None) -> None:
        if new_len is not None:
            self.maxlen = new_len
        [self.update() for i in range(self.maxlen)]

    def sleep(self) -> None:
        sleep(self.dt * (365 * 24 * 3600))

    def run(self) -> None:
        try:
            while True:
                self.update()
                print(self.__last_value, self.__bid[-1], self.__ask[-1])
                self.sleep()
        except KeyboardInterrupt:
            print("Execution stopped manually.")

    def __make_spread(self):
        if self.__spread_model in spread_dict.keys():
            self.compute_spread = spread_dict[self.__spread_model].__get__(self)
        else:
            self.compute_spread = spread_dict['no spread'].__get__(self) 
            print('no spread model loaded')

    @property
    def spread_model(self) -> str:
        return self.__spread_model
    @spread_model.setter
    def spread_model(self, value):
        self.__spread_model = value
        self.reset()


def arithmeticProcess(cls: Type[StochasticProcess]) -> Type[StochasticProcess]:

    def new_value(self: StochasticProcess, prev_value: float,
                  seed: Optional[int] = None) -> float:
        return prev_value + self.new_return(seed)

    cls.new_value = new_value
    return cls
