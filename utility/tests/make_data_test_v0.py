# Import pde pricer class
from deriv_quant_package.pricer.pricing_engine.pricer import pde_pricer
import numpy as np
import pandas as pd

S = 100
K_list = [90, 100, 110]
T_list = [1, 7, 30]
r = 0.03
q = 0.00
vol = 0.1

# Select payoff
option_to_use_list = ["digital_put", "digital_call"]
B = 90
American = False


# PDE parameters
N_x = 200
N_t = 400
matrix_solver = "fast"
boundary = "dirichlet"  # Only supports Dirichlet currently
grid_shift = True

data = pd.DataFrame(
    columns=[
        'vol',
        'Spot',
        'Strike',
        'r',
        'q',
        'mat',
        'option_to_use',
        'premium'])
i = 0
for option_to_use in option_to_use_list:
    for K in K_list:
        for T in T_list:
            pricer = pde_pricer(ref_spot=S,
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
                                grid_shift=grid_shift)
            # Get solutions
            data.loc[i] = [vol, S, K, r, q, T, option_to_use,
                           np.round(pricer.get_price(spot=S, tenor=T), 4)]
            i = i + 1

data.to_csv('test_pde_params.csv')
