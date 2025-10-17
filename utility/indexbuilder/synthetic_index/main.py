from stochastic_process_factory import StochasticProcessFactory
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


factory = StochasticProcessFactory



## The factory can be used to instantiate any of the indices defined in processes_dict.py 

# VolIndex = factory.create_process("DSI30",starting_value) #
## Once a StichasticProcess object is create you an either :
## - call the .make_index() methode to generate a chole chunk of data
## - call the .run() method to get a live stream.
## - mount the tool_kit and call the .run_plot() method to get a live stream plot.

# create process from parameters
from vsi_index import VolatilitySwitchIndex

vol = [0.1,0.5,1.0]
drift = [0,0,0]
T = [15 * 60,10* 60,5*60]
start_val = 10000

VsiIndex = VolatilitySwitchIndex(vol, drift, T, start_val= start_val, spread_model = 'vsi_default')


## ex1 : 
VsiIndex.make_index(10000)
index = np.array(VsiIndex.index)
regimes = np.array(VsiIndex.regimes) 
x = np.linspace(0,1,len(index))

fig = make_subplots()

for regime in np.unique(regimes):
    mask = regimes == regime 
    toplot = mask * index
    toplot[toplot == 0 ] = None
    fig.add_trace(go.Scatter(x=x, y=toplot, name='Stationary regime'))

fig.show()
#             # Add traces
# fig.add_trace(go.Scatter(x=x[mask], y=index[mask], name='Stationary regime'))
# fig.add_trace(go.Scatter(x=x, y=df['Neg'], name='Negative regime'))
# fig.add_trace(go.Scatter(x=x, y=df['Pos'], name='Positive regime'))

# plt.scatter(x, index, c=regimes, cmap='viridis', linestyle='-')

# # plt.plot(index)
# plt.plot(VsiIndex.spread)
# plt.ylim([0.175,0.180])
# plt.figure()
# plt.plot(index)
# plt.figure()
# diff = []
# for i in range((len(index))):
#     diff.append(index[i]/VsiIndex.spread[i])
# plt.plot(diff)
# plt.plot(VsiIndex.ask)

plt.show()

# ##ex 2: need to be manually interupt but would not lose the data
# VolIndex.run()


# ##ex3: need to import the took_kit
# from tool_kit import mount_tool_kit
# mount_tool_kit(VsiIndex)
# VsiIndex.run_plot_spread()

            # Create figure with secondary y-axis


# def plot_index(self, feed, regime, candle='False', resample='1Min'):
#         """
#         Index feed plot with regime
#         :param feed: np.array
#             feed of DSI
#         :param regime: np.array
#             regime of DSI 
#         :param candle: bool     
#             type of graphs normal or candlestick chart
#         """
#         dates = pd.date_range('01/01/2021', periods=len(feed), freq='1s').to_pydatetime()
#         pos, neg = feed * (regime == 1).astype(int), feed * (regime == -1).astype(int)
#         sta = feed * (regime == 0).astype(int)
#         pos[pos == 0], neg[neg == 0], sta[sta == 0] = np.NaN, np.NaN, np.NaN
#         df = pd.DataFrame({'Date': dates, 'Pos': pos, 'Neg': neg, 'Sta': sta})
#         if candle == True:
#             fig = make_subplots(specs=[[{"secondary_y": True}]])
#             df_index_drifted = pd.DataFrame(feed, index=dates).resample(resample).ohlc()
#             df_index_drifted.columns = ['open', 'high', 'low', 'close']
#             fig = go.Figure(data= [go.Candlestick(x=df_index_drifted.index, open=df_index_drifted['open'],
#                             high=df_index_drifted['high'], low=df_index_drifted['low'],
#                             close=df_index_drifted['close'])])
        
#         else :
#             # Create figure with secondary y-axis
#             fig = make_subplots(specs=[[{"secondary_y": True}]])
#             # Add traces
#             fig.add_trace(go.Scatter(x=df['Date'], y=df['Sta'], name='Stationary regime'), secondary_y=False)
#             fig.add_trace(go.Scatter(x=df['Date'], y=df['Neg'], name='Negative regime'), secondary_y=False,)
#             fig.add_trace(go.Scatter(x=df['Date'], y=df['Pos'], name='Positive regime'), secondary_y=False,)
#             # Set y-axes titles
#             fig.update_yaxes(title_text="DSI", secondary_y=False)
#             fig.update_yaxes(title_text="", secondary_y=True)
        
#         # Add figure title
#         fig.update_layout(xaxis={"rangeslider":{"visible":True},"type":"date",
#                                     "range":[df.tail(50)["Date"].min(),df.tail(50)["Date"].max()]},
#                             title_text="Drift Switch index simulation", width=1500, height=700, yaxis=dict(
#        autorange = True,
#        fixedrange= False
#    ))
#         fig.show() 