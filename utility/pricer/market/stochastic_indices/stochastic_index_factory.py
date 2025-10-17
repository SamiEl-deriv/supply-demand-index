from typing import Type
from .stochastic_index_base import StochasticIndex
from .stochastic_index import *

marketsRefDict: dict[str, Type[StochasticIndex]] = {
    "1HZ10V":    {"type": VolIndex,   "params": {"volatility": 0.1,  "yield_rate": 0}},
    "1HZ25V":    {"type": VolIndex,   "params": {"volatility": 0.25, "yield_rate": 0}},
    "1HZ50V":    {"type": VolIndex,   "params": {"volatility": 0.50, "yield_rate": 0}},
    "1HZ75V":    {"type": VolIndex,   "params": {"volatility": 0.75, "yield_rate": 0}},
    "1HZ100V":   {"type": VolIndex,   "params": {"volatility": 1,    "yield_rate": 0}},
    "1HZ200V":   {"type": VolIndex,   "params": {"volatility": 2,    "yield_rate": 0}},
    "1HZ300V":   {"type": VolIndex,   "params": {"volatility": 3,    "yield_rate": 0}},
    "1HZ150V":   {"type": VolIndex,   "params": {"volatility": 1.5,  "yield_rate": 0}},
    "1HZ250V":   {"type": VolIndex,   "params": {"volatility": 2.5,  "yield_rate": 0}},
    "stpRNG":    {"type": StepIndex,  "params": {"step": 0.1}},
    "CRASH1000": {"type": CrashIndex, "params": {"index": 1000}},
    "CRASH500":  {"type": CrashIndex, "params": {"index": 500}},
    "CRASH300":  {"type": CrashIndex, "params": {"index": 300}},
    "BOOM1000":  {"type": BoomIndex,  "params": {"index": 1000}},
    "BOOM500":   {"type": BoomIndex,  "params": {"index": 500}},
    "BOOM300":   {"type": BoomIndex,  "params": {"index": 300}},
    "JD10":      {"type": JumpIndex,  "params": {"volatility": 0.1,  "jump_per_day": 72, "jump_factor": 30}},
    "JD25":      {"type": JumpIndex,  "params": {"volatility": 0.25, "jump_per_day": 72, "jump_factor": 30}},
    "JD50":      {"type": JumpIndex,  "params": {"volatility": 0.5,  "jump_per_day": 72, "jump_factor": 30}},
    "JD75":      {"type": JumpIndex,  "params": {"volatility": 0.75, "jump_per_day": 72, "jump_factor": 30}},
    "JD100":     {"type": JumpIndex,  "params": {"volatility": 1,    "jump_per_day": 72, "jump_factor": 30}},
    "DEX900DN":  {"type": DEXIndex,   "params": {"volatility": 0.25, "interest_rate": 0, "jump_frq": 20*365*24, "proba_jump_up": 0.8, "jump_up_size": 4e-4, "jump_down_size": 3e-3}},
    "DEX900UP":  {"type": DEXIndex,   "params": {"volatility": 0.25, "interest_rate": 0, "jump_frq": 20*365*24, "proba_jump_up": 0.2, "jump_up_size": 3e-3, "jump_down_size": 4e-4}}
}


class StochasticIndexFactory:
    """
    stochastic index factory class
    """

    _builders = marketsRefDict

    @classmethod
    def CreateStochIndex(cls, marketId: str):
        """
        Creates a new stochastic index using a stored builder

        Parameters
        ----------
            IndexId : str
                The stochastic index ID for the builder
            **kwargs
                arguments of the stochastic index constructor -> float

        Returns
        -------
        StochasticIndex
            StochasticIndex class instance

        """

        model = cls._builders.get(marketId)
        if not model:
            raise ValueError(f"No index ID assigned {marketId}")
        index_type = model.get("type")
        index_params = model.get("params")
        return index_type(**index_params)
