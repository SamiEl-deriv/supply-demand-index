# %%

import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WXAgg', depending on your system
from stochastic_process_factory import StochasticProcessFactory

# Instantiate StochasticProcessFactory
factory = StochasticProcessFactory

# Create instances of stochastic processes
Index = factory.create_process("VSI_new", 1000)

# Generate 1 million simulated data points
simulated_data = []
for _ in range(10000):
    Index.update()  # or VolIndex.make_index() depending on your requirement
    simulated_data.append(Index.index[-1])  # Append the latest index value to the list

# Plot the simulated data
plt.plot(simulated_data)
plt.xlabel('Time')
plt.ylabel('Index Value')
plt.title('Simulated Index')
plt.grid(True)
plt.show()

# -------------------
# %%
import plotly.graph_objects as go
from stochastic_process_factory import StochasticProcessFactory

# Instantiate StochasticProcessFactory
factory = StochasticProcessFactory

# Create instances of stochastic processes
VolIndex = factory.create_process("VSI_new", 100)

# Generate 1 million simulated data points
simulated_data = []
for _ in range(100000):
    VolIndex.update()  # or VolIndex.make_index() depending on your requirement
    simulated_data.append(VolIndex.index[-1])  # Append the latest index value to the list

# Create a Plotly figure
fig = go.Figure()

# Add a scatter plot of the simulated data
fig.add_trace(go.Scatter(x=list(range(len(simulated_data))), y=simulated_data, mode='lines', name='Simulated Index Data'))

# Update layout with titles and axis labels
fig.update_layout(
    title='Simulated Index Data',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Index Value'),
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)'  # Make plot background transparent
)

# Show the interactive plot
fig.show()

# %%
