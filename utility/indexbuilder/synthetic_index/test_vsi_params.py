import datetime
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vsi_index import VolatilitySwitchIndex

# vol = [0.1,0.5,1.0]
drift = [0,0,0]
# T = [30*60, 30*60, 30*60]
start_val = 10000

param_test  = { #'test1' : [[0.1,0.5,1.0],[30*60, 30*60, 30*60]],
                # 'test2' : [[0.25,1.0,2.0],[30*60, 30*60, 30*60]],
                # 'test3' : [[0.25,1.0,2.0],[15*60, 15*60, 15*60]],
                'test4' : [[0.1,0.5,1.0],[60*60, 30*60, 10*60]]#,
                # 'test5' : [[0.5,0.75,1.5],[60*60, 30*60, 10*60]],
                # 'test6' : [[0.5,0.75,1.5],[30*60, 15*60, 5*60]]
}
lens = [60*60*3]#, 60*60*12, 60*60*24]  
for test in param_test:
    vol = param_test[test][0]
    T = param_test[test][1]
    for index_len in lens :
        VsiIndex = VolatilitySwitchIndex(vol, drift, T, start_val= start_val, spread_model = 'vsi_default')

        VsiIndex.make_index(index_len)
        index = np.array(VsiIndex.index)
        regimes = np.array(VsiIndex.regimes) 

        x = np.linspace(0,1,index_len)
        start_date = datetime.datetime(2024, 1, 1)
        date_list = [start_date + datetime.timedelta(seconds = i) for i in range(index_len)]

        fig = make_subplots()

        for regime in np.unique(regimes):
            mask = regimes == regime 
            toplot = mask * index
            toplot[toplot == 0 ] = None
            fig.add_trace(go.Scatter(x=date_list, y=toplot, name='vol= {}%, T= {}min'.format(VsiIndex.volatility[regime]*100,VsiIndex.T[regime]/60)))

        fig.show()
