from stochastic_process_base import StochasticProcess
from range_break_index import RangeBreakIndex
from bear_bull_index import BearBullIndex
from crash_boom_index import CrashIndex, BoomIndex
from dex_index import DexIndex
from dsi_index import DriftSwitchIndex
from jump_index import JumpIndex
from step_index import StepIndex
from volatility_index import VolatilityIndex
from vsi_index import VolatilitySwitchIndex
from vsi_index_new import VolatilitySwitchIndex_new
from typing import Type


dt = 1/(365 * 24 * 3600)


marketsRefDict: dict[str, Type[StochasticProcess]] = {
    "1HZ10V":    {"type": VolatilityIndex,   "params": {"volatility": 0.1}},
    "1HZ25V":    {"type": VolatilityIndex,   "params": {"volatility": 0.25}},
    "1HZ50V":    {"type": VolatilityIndex,   "params": {"volatility": 0.50}},
    "1HZ75V":    {"type": VolatilityIndex,   "params": {"volatility": 0.75}},
    "1HZ100V":   {"type": VolatilityIndex,   "params": {"volatility": 1}},
    "1HZ200V":   {"type": VolatilityIndex,   "params": {"volatility": 2}},
    "1HZ300V":   {"type": VolatilityIndex,   "params": {"volatility": 3}},
    "1HZ150V":   {"type": VolatilityIndex,   "params": {"volatility": 1.5}},
    "1HZ250V":   {"type": VolatilityIndex,   "params": {"volatility": 2.5}},

    "stpRNG":    {"type": StepIndex,  "params": {"step_size": 0.1}},

    "CRASH1000": {"type": CrashIndex, "params": {"mean_interval": 1000, "diff_percent": 0.001, "mdt":-5.619}},
    "CRASH500":  {"type": CrashIndex, "params": {"mean_interval": 500, "diff_percent": 0.002, "mdt":-5.619}},
    "CRASH300":  {"type": CrashIndex, "params": {"mean_interval": 300, "diff_percent": 0.003, "mdt":-14.619814}},
    "BOOM1000":  {"type": BoomIndex,  "params": {"mean_interval": 1000, "diff_percent": 0.001, "mut":5.619}},
    "BOOM500":   {"type": BoomIndex,  "params": {"mean_interval": 500, "diff_percent": 0.002, "mut":5.619}},
    "BOOM300":   {"type": BoomIndex,  "params": {"mean_interval": 300, "diff_percent": 0.003, "mut":14.619814}},

    "JD10":      {"type": JumpIndex,  "params": {"volatility": 0.1,  "jump_factor": 30}},
    "JD25":      {"type": JumpIndex,  "params": {"volatility": 0.25, "jump_factor": 30}},
    "JD50":      {"type": JumpIndex,  "params": {"volatility": 0.5,  "jump_factor": 30}},
    "JD75":      {"type": JumpIndex,  "params": {"volatility": 0.75, "jump_factor": 30}},
    "JD100":     {"type": JumpIndex,  "params": {"volatility": 1,    "jump_factor": 30}},

    "DEX900DN":  {"type": DexIndex,   "params": {"volatility": 0.25, "interest_rate": 0, "jump_frq": 20*365*24, "proba_jump_up": 0.8, "jump_up_size": 4e-4, "jump_down_size": 3e-3}},
    "DEX900UP":  {"type": DexIndex,   "params": {"volatility": 0.25, "interest_rate": 0, "jump_frq": 20*365*24, "proba_jump_up": 0.2, "jump_up_size": 3e-3, "jump_down_size": 4e-4}},

    "RB100":    {"type": RangeBreakIndex, "params": {"step_size": 1, "perc_out": 0.01,  "jump_param": 0.5, "wait_time": 900}},
    "RB200":    {"type": RangeBreakIndex, "params": {"step_size": 1, "perc_out": 0.005, "jump_param": 0.5, "wait_time": 1800}},

    "DSI10":    {"type": DriftSwitchIndex, "params": {"drift": 100, "volatility": 0.1, "gamma": 0.4980997907, "regime_duration": 10 * 60}},
    "DSI20":    {"type": DriftSwitchIndex, "params": {"drift": 60,  "volatility": 0.1, "gamma": 0.4977183219, "regime_duration": 20 * 60}},
    "DSI30":    {"type": DriftSwitchIndex, "params": {"drift": 35,  "volatility": 0.1, "gamma": 0.4980031156, "regime_duration": 30 * 60}},

    # ----------
    "VSI":      {"type": VolatilitySwitchIndex, "params": {"vol":[0.1, 0.5, 0.75], "drift":[0, 0, 0], "T":[3,4,5], "zeta":[0.5, 0.3, 0.2], "P":[[0.5, 0.4, 0.3],[0.3, 0.4, 0.3],[0.2, 0.2, 0.4]]}},  #   Column sum = 1
    "VSI_new":      {"type": VolatilitySwitchIndex_new, "params": {"vol":[0.1, 0.5, 0.75], "drift":[0, 0, 0], "T":[3,4,5], "zeta":[0.5, 0.3, 0.2], "P":[[0.5, 0.2, 0.3],[0.3, 0.4, 0.3],[0.1, 0.2, 0.7]]}}   # Row sum = 1
}
