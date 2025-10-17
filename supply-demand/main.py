import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from fbm import fbm
import matplotlib.ticker as ticker
from scipy.stats import norm
from scipy.signal import lfilter


def generate_gbm_path(
    S_0: float,
    mu: float,
    sigma: float,
    duration_in_seconds: int,
    dt: float = 1 / (86_400 * 365),
    random_seed: int | None = None,
) -> npt.NDArray[np.float64]:
    if random_seed is not None:
        np.random.seed(random_seed)

    num_second_per_tick = int(1 / dt / (86_400 * 365))
    num_step = int(duration_in_seconds / num_second_per_tick)
    price_path = (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * np.random.normal(
        0, 1, size=num_step
    )
    return S_0 * np.exp(np.cumsum(price_path))


sigmas = [0.1, 0.25, 0.5, 0.75, 1.00]
tolerance = 1e-3  # Allow small error margin

for sigma in sigmas:
    S = generate_gbm_path(
        S_0=1_000_000,
        mu=0,
        sigma=sigma,
        duration_in_seconds=86_400 * 30,
    )
    log_returns = np.diff(np.log(S))
    estimated_sigma = np.std(log_returns) * np.sqrt(86_400 * 365)
    assert np.isclose(estimated_sigma, sigma, atol=tolerance), (
        f"Sigma check failed: expected {sigma}, got {estimated_sigma:.4f}"
    )
    print(f"Sigma {sigma:.2f}: PASS (Estimated {estimated_sigma:.4f})")

print("All sigma checks passed successfully! 🎉")


def compute_expected_log_returns_from_mu(
    mu: float, sigma: float, time_delta_in_years: float
) -> float:
    return time_delta_in_years * (mu - sigma**2 / 2)


def compute_mu_from_expected_log_returns(
    expected_log_returns: float, sigma: float, time_delta_in_years: float
) -> float:
    return expected_log_returns / time_delta_in_years + sigma**2 / 2


mu = 35
sigma = 1
duration_in_seconds = 3_600
time_delta_in_years = duration_in_seconds / (86_400 * 365)
expected = compute_expected_log_returns_from_mu(
    mu=mu, sigma=sigma, time_delta_in_years=time_delta_in_years
)

num_samples = 100_000
S_0 = 100_000
log_returns = []
for _ in range(num_samples):
    S = generate_gbm_path(
        S_0=S_0,
        mu=mu,
        sigma=sigma,
        duration_in_seconds=duration_in_seconds,
    )
    log_returns.append(np.log(S[-1] / S_0))
simulated_mean = np.mean(log_returns)

tolerance = 1e-4  # adjust depending on your acceptable error

# Use np.isclose for the assertion
assert np.isclose(simulated_mean, expected, atol=tolerance), (
    f"Mean log return check failed: expected {expected:.6f}, got {simulated_mean:.6f}"
)

print(f"Expected {expected:.6f}: PASS (Simulated {simulated_mean:.6f})")

print("Assertion passed: Simulated mean log-return is close to expected.")


# ---- Parameters ----
S_0 = 100_000
sigma = 0.1  # Fixed sigma as per your request
duration_in_seconds = 3_600  # 1 hour
dt = 1 / (86_400 * 365)
mu_values = [5, 10, 20, 50]  # Example mu, change as you like
num_paths = 1000
base_seed = 2025

fig, axs = plt.subplots(1, len(mu_values), figsize=(22, 5), sharey=True, dpi=200)

for idx, mu in enumerate(mu_values):
    real_sigmas = []
    real_drifts = []
    real_mus = []
    all_paths = []

    for i in range(num_paths):
        gbm_path = generate_gbm_path(
            S_0=S_0,
            mu=mu,
            sigma=sigma,
            duration_in_seconds=duration_in_seconds,
            dt=dt,
            random_seed=base_seed + i,
        )
        all_paths.append(gbm_path)
        axs[idx].plot(gbm_path, color="tab:blue", alpha=0.05, linewidth=0.5)
        log_returns = np.diff(np.log(gbm_path))
        this_sigma = log_returns.std(ddof=1) / np.sqrt(dt)
        real_sigmas.append(this_sigma)
        this_drift = log_returns.mean() / dt
        real_drifts.append(this_drift)
        this_mu = this_drift + 0.5 * this_sigma**2
        real_mus.append(this_mu)
    all_paths_np = np.vstack(all_paths)
    mean_path = np.mean(all_paths_np, axis=0)
    axs[idx].plot(mean_path, color="red", lw=2, label="Mean Price")

    axs[idx].set_title(
        f"$\\mu$ = {mu:0.2%}\n"
        f"Realised: $\\mu$={np.mean(real_mus):.2%}, "
        f"$\\sigma$={np.mean(real_sigmas):.2%}, drift={np.mean(real_drifts):.2%}"
    )
    axs[idx].set_xlabel("Tick (seconds)")
    if idx == 0:
        axs[idx].set_ylabel("Price")
    axs[idx].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useOffset=False, useMathText=False)
    )
    axs[idx].ticklabel_format(style="plain", axis="y")
    axs[idx].legend(loc="upper left", fontsize="small")
    axs[idx].grid(True)

fig.suptitle(
    f"GBM simulated price paths ({num_paths:,} paths)\n(Inputs: $\\sigma$={sigma:.2%})",
    fontsize=20,
    weight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("gbm_simulation.png", dpi=300)
plt.show()


def compute_mu_from_probability_itm(P, S, K, sigma, T, option_type="call"):
    """
    Solves for mu in GBM given desired probability P ending
    in-the-money (decimal), S (spot), K (strike),
    sigma (vol, annualized), T (years), and
    option_type: 'call' or 'put'
    """
    if option_type == "call":
        d2 = norm.ppf(P)  # For calls, N(d2) = P
    elif option_type == "put":
        d2 = -norm.ppf(P)  # For puts, N(-d2) = P => d2 = -invNorm(P)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    mu = (d2 * sigma * np.sqrt(T) - np.log(S / K)) / T + 0.5 * sigma**2
    return mu


# Example usage
P = 0.5  # e.g., 80% chance in the money
S = 100_000  # Spot price
K = 100_000  # Strike price
sigma = 1  # Volatility
T = 3_600 / (86_400 * 365)  # Time to expiry in years

mu_call = compute_mu_from_probability_itm(P, S, K, sigma, T, option_type="call")
mu_put = compute_mu_from_probability_itm(P, S, K, sigma, T, option_type="put")
print("Call mu:", mu_call)
print("Put  mu:", mu_put)


### Example
# Parameters
n = 3_600  # Number of time steps
mu = 3  # Example constant drift per step (set this as f(net_exposure) if you wish)
phi = 0.6  # AR(1) coefficient (>0 for trend amplification)
sigma_target = 0.1  # Desired process (return) volatility
dt = 1 / (365 * 86_400)

# Scale innovation volatility for AR(1)
sigma_innov = sigma_target * np.sqrt(1 - phi**2)

returns = np.zeros(n)
epsilon = np.random.normal(0, sigma_innov, n) * np.sqrt(dt)

for t in range(1, n):
    returns[t] = mu * dt + phi * returns[t - 1] + epsilon[t]

# Get price path from returns
price = 100_000 * np.exp(np.cumsum(returns))  # Start price = 100

# Plot
plt.figure(figsize=(10, 4))
plt.plot(price)
plt.title(f"Simulated AR(1) GBM price path (phi={phi}, mu={mu}, sigma={sigma_target})")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.grid()
plt.savefig("autoregressive_gbm.png", dpi=300)
plt.show()


import numpy as np
from scipy.signal import lfilter


def generate_ar1_price_path(
    S_0: float,
    mu: float | np.ndarray,
    phi: float,
    sigma: float,
    duration_in_seconds: int,
    dt: float = 1 / (86_400 * 365),
    gbm_drift: bool = True,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a price series following an AR(1) trend/momentum process for log-returns.

    Parameters
    ----------
    S_0 : float
        Initial price.
    mu : float or np.ndarray
        Drift per unit time for price process. If gbm_drift=True, log-return drift uses (mu-0.5*sigma^2).
        Can be scalar or array of length n_steps.
    phi : float
        AR(1) coefficient.
    sigma : float
        Target standard deviation (volatility) of returns, per unit time.
    duration_in_seconds : int
        Total simulation duration, in seconds.
    dt : float, optional
        Time step size (default: one second in "years").
    gbm_drift : bool, optional
        If True, use (mu-0.5*sigma^2) for log-return drift (classical GBM convention).
        If False, use mu as the *log-return* drift directly.
    random_seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    price : np.ndarray
        Simulated price series.
    returns : np.ndarray
        Simulated log-returns.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Determine number of steps for given duration and dt
    num_second_per_tick = int(1 / dt / (86_400 * 365))
    num_step = int(duration_in_seconds / num_second_per_tick)

    mu = np.asarray(mu)
    if mu.size == 1:
        mu_vec = np.full(num_step, mu)
    elif mu.size == num_step:
        mu_vec = mu
    else:
        raise ValueError("mu must be a scalar or have length n_steps.")

    # Ito/GBM correction
    if gbm_drift:
        drift_vec = mu_vec - 0.5 * sigma**2
    else:
        drift_vec = mu_vec

    # Correct innovation volatility for AR(1) autocorrelation
    sigma_innov = sigma * np.sqrt(1 - phi**2)
    epsilon = np.random.normal(0, sigma_innov, num_step) * np.sqrt(dt)
    total_input = drift_vec * dt + epsilon

    # Vectorized AR(1) solution
    returns = lfilter([1], [1, -phi], total_input)
    price = S_0 * np.exp(np.cumsum(returns))
    return price, returns


ar1_price_params = {
    "S_0": 100_000,
    "mu": 0.0,
    "phi": 0,
    "sigma": 0.1,
    "duration_in_seconds": 864_000,
    "random_seed": int(np.random.randint(0, 1000, 1)[0]),
}
dt = 1 / (86_400 * 365)
sigma = ar1_price_params["sigma"]

ar1_price_path = generate_ar1_price_path(**ar1_price_params)[0]

# Calculate realized volatility (annualized)
log_rets = np.diff(np.log(ar1_price_path))
simulated_vol = log_rets.std(ddof=1) / np.sqrt(dt)

expected_vol = sigma  # target volatility from input
tolerance = (
    0.003  # Acceptable tolerance (~0.3% for large samples; adjust to your liking)
)

assert np.isclose(simulated_vol, expected_vol, atol=tolerance), (
    f"Volatility check failed: expected {expected_vol:.6f}, got {simulated_vol:.6f}"
)

print(f"Expected volatility {expected_vol:.6f}: PASS (Simulated {simulated_vol:.6f})")
print("Assertion passed: Simulated volatility is close to expected.")


# ---- Parameters ----
S_0 = 100_000
sigma = 0.5
mu = 10
duration_in_seconds = 3_600  # 1 hour
base_seed = 2024
dt = 1 / (86_400 * 365)
phi_values = [0, -0.2, -0.5, -0.9]
num_paths = 1000

fig, axs = plt.subplots(1, len(phi_values), figsize=(22, 5), sharey=True, dpi=200)

for idx, phi in enumerate(phi_values):
    real_sigmas = []
    real_drifts = []
    real_mus = []
    all_paths = []

    for i in range(num_paths):
        # Update random seed for each path for reproducibility
        ar1_path, _ = generate_ar1_price_path(
            S_0=S_0,
            mu=mu,
            phi=phi,
            sigma=sigma,
            duration_in_seconds=duration_in_seconds,
            dt=dt,
            random_seed=base_seed + i,
        )
        all_paths.append(ar1_path)
        # Plot path (thin lines, alpha for visibility)
        axs[idx].plot(ar1_path, color="tab:blue", alpha=0.05, linewidth=0.5)
        log_returns = np.diff(np.log(ar1_path))
        this_sigma = log_returns.std(ddof=1) / np.sqrt(dt)
        real_sigmas.append(this_sigma)
        this_drift = log_returns.mean() / dt
        real_drifts.append(this_drift)
        this_mu = this_drift + 0.5 * this_sigma**2
        real_mus.append(this_mu)
    all_paths_np = np.vstack(all_paths)
    mean_path = np.mean(all_paths_np, axis=0)
    axs[idx].plot(mean_path, color="red", lw=2, label="Mean Price")

    axs[idx].set_title(
        f"$\\phi$ = {phi}\n"
        f"Realised: $\\mu$={np.mean(real_mus):.2%}, "
        f"$\\sigma$={np.mean(real_sigmas):.2%}, drift={np.mean(real_drifts):.2%}"
    )
    axs[idx].set_xlabel("Tick (seconds)")
    if idx == 0:
        axs[idx].set_ylabel("Price")
    axs[idx].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useOffset=False, useMathText=False)
    )
    axs[idx].ticklabel_format(style="plain", axis="y")
    axs[idx].legend(loc="upper left", fontsize="small")
    axs[idx].grid(True)

fig.suptitle(
    f"AR(1) simulated price paths ({num_paths:,} paths)\n"
    f"(Inputs: $\\mu$={mu:.2%}, $\\sigma$={sigma:.2%}, drift={mu - sigma**2 / 2:.2%}, "
    f"$S_0$={S_0:,.0f})",
    fontsize=20,
    weight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("ar1_simulation.png", dpi=300)
plt.show()


def generate_fgbm_path(
    S_0: float,
    mu: float,
    sigma: float,
    duration_in_seconds: int,
    hurst: float,
    random_seed: int = None,
) -> np.ndarray:
    N = duration_in_seconds
    dt_years = 1 / (86400 * 365)
    times = np.arange(N + 1) * dt_years
    T_years = N * dt_years
    if random_seed is not None:
        np.random.seed(random_seed)
    fbm_path = fbm(n=N, hurst=hurst, length=T_years, method="daviesharte")
    drift = (mu - 0.5 * sigma**2) * times
    log_path = drift + sigma * fbm_path
    return S_0 * np.exp(log_path)


S_0 = 100_000
sigma = 0.1
mu_factor = 500
mu = sigma * mu_factor
duration_in_seconds = 3600  # 1 hour
base_seed = 2024  # fixed for reproducibility
dt = 1 / (86_400 * 365)
H_values = [0.45, 0.5, 0.55, 0.6]  # Hurst exponent
num_paths = 1_000

fig, axs = plt.subplots(1, len(H_values), figsize=(24, 6), sharey=True, dpi=300)

# max_plot_paths = 30  # Only plot this many paths per subplot

for idx, H in enumerate(H_values):
    real_sigmas = []
    real_drifts = []  # empirical drift, d = mean(log returns)/dt
    real_mus = []  # gbm drift, $\mu = d + 0.5 \times (\text{realized vol})^2$

    all_paths = []  # Store each path for mean calculation

    for i in range(num_paths):
        path = generate_fgbm_path(
            S_0, mu, sigma, duration_in_seconds, hurst=H, random_seed=base_seed + i
        )
        all_paths.append(path)
        # You may choose to thin the lines for plotting speed and visibility

        axs[idx].plot(path, color="tab:blue", alpha=0.05, linewidth=0.5)
        # Calculate and store realized volatility (use Bessel's correction ddof=1)
        log_returns = np.diff(np.log(path))
        ####
        this_sigma = log_returns.std(ddof=1) / np.sqrt(dt)
        real_sigmas.append(this_sigma)
        ####
        this_drift = np.mean(log_returns) / dt
        real_drifts.append(this_drift)
        ####
        this_mu = this_drift + 0.5 * this_sigma**2
        real_mus.append(this_mu)

    # Calculate and plot the mean price at each tick across all paths
    all_paths_np = np.vstack(all_paths)
    mean_path = np.mean(all_paths_np, axis=0)
    axs[idx].plot(mean_path, color="red", lw=2, label="Mean Price")

    # Show mean realized volatility in subplot title
    axs[idx].set_title(
        f"H = {H}\n"
        f"Realised: $\\mu$={np.mean(real_mus):.2%}, "
        f"$\\sigma$={np.mean(real_sigmas):.2%}, drift={np.mean(real_drifts):.2%}"
    )
    axs[idx].set_xlabel("Tick (seconds)")
    if idx == 0:
        axs[idx].set_ylabel("Price")
    axs[idx].yaxis.set_major_formatter(
        ticker.ScalarFormatter(useOffset=False, useMathText=False)
    )
    axs[idx].ticklabel_format(style="plain", axis="y")
    axs[idx].legend(loc="upper left", fontsize="small")
    axs[idx].grid(True)

fig.suptitle(
    f"Fractional GBM ({num_paths:,} paths)\n"
    f"(Inputs: $\\mu$={mu:.2%}, $\\sigma$={sigma:.2%}, drift={(mu - sigma**2 / 2):.2%})",
    fontsize=20,
    weight="bold",
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("fractional_gbm.png", dpi=300)
plt.show()
