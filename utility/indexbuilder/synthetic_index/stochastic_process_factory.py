from stochastic_process_base import StochasticProcess
from processes_dict import marketsRefDict
from typing import Type, Dict, Optional

class StochasticProcessFactory:

    __builders: Dict[str, Type[StochasticProcess]] = marketsRefDict

    @classmethod
    def register_type(cls, process_name: str, process_params: Dict) -> None:
        cls.__builders[process_name] = process_params

    @classmethod
    def create_process(cls, process_name: str, start_val, dt  = 1/(365 * 24 * 3600)) -> StochasticProcess:
        return cls.__builders[process_name]['type'](**cls.__builders[process_name]['params'],start_val= start_val, dt=dt)

