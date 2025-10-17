from stochastic_process_factory import StochasticProcessFactory
import matplotlib.pyplot as plt
from volatility_index import VolatilityIndex
from vsi_index import VolatilitySwitchIndex


factory = StochasticProcessFactory

## The factory can be used to instantiate any of the indices defined in processes_dict.py 
starting_value = 10000

VolIndex = VolatilitySwitchIndex(vol=[0.1,2.0], T=[10.0,10.0],start_val=1000, spread_model = 'vsi_default') #, spread_param = {'percentage'  : 50})#, spread_model = 'percentage')

# ## ex1 : 
# VolIndex.make_index(100)
# index = VolIndex.index
# plt.plot(index)
# plt.show()

##ex 2: need to be manually interupt but would not lose the data
# VolIndex.run()


# ##ex3: need to import the took_kit
from tool_kit import mount_tool_kit
mount_tool_kit(VolIndex)
VolIndex.run_plot()