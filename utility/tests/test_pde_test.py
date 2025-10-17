# Import pde pricer class
from deriv_quant_package.pricer.pricing_engine.pricer import pde_pricer
import numpy as np
import pandas as pd
import pytest

data = pd.read_csv('deriv_quant_package/tests/test_pde_params.csv')

# PDE parameters
N_x = 200
N_t = 400
scheme_to_use = "crank_nicolson_rannacher"
matrix_solver = "fast"
boundary = "dirichlet"  # Only supports Dirichlet currently
grid_shift = True
American = False

test_parameters = [tuple(data.loc[i]) for i in data.index.values]


@pytest.mark.parametrize(
    "i, vol, S, K, r, q, T, option_to_use, premium_expected",
    test_parameters)
def test_multiple(i, vol, S, K, r, q, T, option_to_use, premium_expected):
    pricer = pde_pricer(
        ref_spot=S,
        strike=K,
        riskless_rates=r,
        dividend_rates=q,
        time_to_expiry=T,
        vols=vol,
        option=option_to_use,
        barrier=K,
        american=American,
        N_S=N_x,
        N_t=N_t,
        solver=matrix_solver,
        boundary=boundary,
        grid_shift=grid_shift)  # Optional PDE arguments

    assert np.round(pricer.get_price(spot=S, tenor=T), 4) == premium_expected
