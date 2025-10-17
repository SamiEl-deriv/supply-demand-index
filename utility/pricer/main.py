# Import pde pricer class
from deriv_quant_package.pricer.pricer import pde_pricer
# Import visualization classes
import matplotlib.pyplot as plt
# Import timer class
import time
#import numpy
import numpy as np
#import pandas
import pandas as pd

#loading implied volatility surface
# ..._volsurface.csv must be a n*3 array containing n
# triplets (T_i, k_i, sigma(T_i,k_i))
# Where : - T_i is the time to maturu-ity in years
#         - k_i is the moneyness 
#         - sigma(T_i,k_i) is the volatility at (T_i, k_i)

#loading the file in a panda dataframe
df = pd.read_csv(r"XAUUSD_volsurface.csv",sep = "\t",header=None)
#converting to np.array
volatility_quotations = df.to_numpy()

# data processing (converting to the right format)
volatility_quotations = volatility_quotations.astype(str)
volatility_quotations = np.char.replace(volatility_quotations, ",", ".")
volatility_quotations = volatility_quotations.astype(float)

# more data processing (taking values for T<1 years)
volatility_quotations = volatility_quotations[volatility_quotations[:,0]<1.1]
volatility_quotations = volatility_quotations[volatility_quotations[:,0]>1/365]
# Vol is in percentage --> converting in a number betweeen 0 and 1
volatility_quotations[:,2]=volatility_quotations[:,2]/100

# A function that converts this array into a dictionnarry
# that has the right format for the pricing engine
def convert_array_to_dict(volatility_quotations):
    tenors = np.unique(volatility_quotations[:,0])
    vol_dict = {}
    for tenor in tenors:
        vol_dict[tenor]={}
        indexes = np.where(volatility_quotations[:,0]==tenor)
        for index in indexes[0]:
            vol_dict[tenor][volatility_quotations[index,1]]= volatility_quotations[index,2]
    return vol_dict

#converting our array into a dictionnary using the function
vol_dict = convert_array_to_dict(volatility_quotations)

###Pricing###

# Option parameters
S = 1
K = 1
r = 0.01
q = 0.0
T = 1
vol = vol_dict
American = False
smoothing = 0.0000
min_value_for_local_vol = 0.1
gap_with_existing_points = 0.05
gap_between_new_points = 0.025
number_new_points = 10

# Select payoff
option_to_use = "up_out_call"
B = 120

option_to_use = "sharkfinKO_call"
B = 1.20

# PDE parameters
N_x = 600
N_t = 600
scheme_to_use = "crank_nicolson_rannacher"
matrix_solver = "fast"
boundary = "dirichlet"  # Only supports Dirichlet currently
grid_shift = True


"""
Activate and run PDE pricer
"""

# Start timer
start = time.perf_counter_ns()

pricer = pde_pricer(
    ref_spot=S,
    strike=K,
    riskless_rates=r,
    dividend_rates=q,
    time_to_expiry=T,
    vols=vol,
    option=option_to_use,
    barrier=B,
    american=American,
    N_S=N_x,
    N_t=N_t,
    solver=matrix_solver,
    boundary=boundary,
    grid_shift=grid_shift,
    volsurface_type='strike',
    volsurface_interp='TPS_local_vol',
    smoothing=smoothing,
    min_value_for_local_vol=min_value_for_local_vol,
    gap_with_existing_points =gap_with_existing_points,
    gap_between_new_points = gap_between_new_points,
    number_new_points = number_new_points)  # Optional PDE arguments

# Get solutions
price_surface = pricer.get_price_surface(with_coords=True)
solution = pricer.get_price_curve(with_coords=True)

# Price specific point
spot = 1
tenor = 1
price = pricer.get_price(spot=spot, tenor=tenor)

stop = time.perf_counter_ns()

# i = S-axis, j = t-axis, k = V-axis
meshS = price_surface[:, :, 0]
mesht = price_surface[:, :, 1]
meshV = price_surface[:, :, 2]

# i = S-axis, j = V-axis
curveS = solution[:, 0]
curveV = solution[:, 1]

# Debugging prints
option_name = option_to_use.replace("_", " ").title()

print(f"\nOption payoff: {option_to_use}\nScheme: {scheme_to_use}\nMode: {matrix_solver}\n\n{N_x=}\n{N_t=}\nBoundary: {boundary}\n\nElapsed time: {(stop - start) * 1e-9} seconds")
print(f"{option_name} contract with {spot=}, {tenor=} has {price=}")


"""
Plots
"""
plt.close()

title_substring = f"{'American' if American else 'European'} {option_name} "

# Plot t-slices
for i in range(meshS.shape[1]):
    # if i % 10 == 0:
    plt.plot(meshS[:, i], meshV[:, i])
plt.title(title_substring + "Option Prices")
plt.xlabel("S")
plt.ylabel("V")
plt.draw()

# Plot lower spot boundary conditions
plt.figure()
plt.plot(mesht[0, :], meshV[0, :])
plt.title(title_substring + "Lower Spot Boundary Conditions")
plt.xlabel("t")
plt.ylabel("V")
plt.draw()

# Plot upper spot boundary conditions
plt.figure()
plt.plot(mesht[-1, :], meshV[-1, :])
plt.title(title_substring + "Upper Spot Boundary Conditions")
plt.xlabel("t")
plt.ylabel("V")
plt.draw()

# Plot t=0 solution
plt.figure()
plt.plot(meshS[:, 0], meshV[:, 0])
plt.title(title_substring + "Option Prices")
plt.xlabel("S")
plt.ylabel("V")
plt.draw()

# Plot T_max boundary conditions (payoff)
plt.figure()
plt.plot(meshS[:, -1], meshV[:, -1])
plt.title(title_substring + "Initial Conditions")
plt.xlabel("S")
plt.ylabel("V")
plt.draw()

# Plot 3d surface of option prices
fig = plt.figure()
ax = plt.axes(projection='3d', computed_zorder=False)

ax.plot_surface(meshS, mesht, meshV, rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
# We must cast tenor to T - tenor as the t-axis represents the starting
# time of the contract
ax.scatter(spot, T - tenor, price, c="#00ff00", alpha=1, depthshade=False)

ax.set_title(title_substring + "Option Prices")
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')
plt.draw()

plt.show()
