# Tactical Documentation (WIP)

### Squad - Derived Indices

#### Last Update: 13 March 2025

## Product Information

### Product Description


The Tactical Index is an index derived from a portfolio containing long/short positions on a _universe_ of underlyings and cash. Depending on the underlying technical indicator/strategy, the customer can bet on more complex behaviour of the underlying (such as reversals or trends).

The underlying may range from any index, from finacial (crypto, FX & commodities) or synthetic indices (Volatility Indices). The currently implemented indicators used to generate the signals currently range across the following:

1. RSI.

Planned indices or indices under ongoing development:
1. MA Crossover;
2. Bollinger Bands;
3. MACD;
4. Pairs Trading.

The behaviour of these indices is highly dependent on the indicator used (in turn dependent on its parameters), the signal function and the behaviour of the underlying. Since a number of these strategies depend on some rolling window, there is no index generation for a pre-defined initial lookback period (`no_lb`). The index is generated every time a new underlying tick is received, whereas the portfolio is rebalanced, i.e the signal direction is selected, over a regular period called the `rebalancing_period`. 

Tweaking the index parameters, we can drastically change the behaviour of each index. Five standard configurations have been used, each capitalizing on a particular viewpoint:

* Rebound Indices (e.g: RSIXAGCL)
  * Mean-reverting behaviour, with a bias on long positions
* Pullback Indices (e.g: RSIXAGCS)
  * Mean-reverting behaviour, with a bias on short positions
* Trend Up Indices (e.g: RSIXAGML)
  * Trend-following behaviour, with a bias on long positions
* Trend Down Indices (e.g: RSIXAGMS)
  * Trend-following behaviour, with a bias on short positions
* Standard Indices (e.g: BBXAUST)
  * Balanced behaviour, not biased to any particular position

If none of the above are applicable, we may just postfix with a number instead, e.g:

* MACOXAG1

This may be subject to change until more details are confirmed, however.

### Changing Parameters
The parameters of the indices above may be set via a BO tool, allowing us to change the behaviour of the indices in case of an emergency or to better reposition ourselves against current trends on the underlying. To access the BO tool, select the Feed Configuration Tool after logging into BackOffice then scroll down until the Tactical section. All parameters may be changed there, but it requires access via a Dual Control Code (DCC), where a member from both the Structuring and Dealing team with write access are required to change the parameters.

In the report onwards, we will refer to the indices with the aliases in the brackets, as they are named in BE & Backoffice.

### Index Construction

For a guide on parameter configurations for mvlib's Tactical Index generators, refer to `Config Documentation.md`.

#### Single Underlying

The spot price at the $n$-th tick, $S_n$ is defined based on the $n$-th received underlying quote (spot) $U_n$:

$$
S_{n+1} = S_n \frac{1+\omega^L_n r_n}{1+\omega^S_n r_n}
$$

Where:

* $\omega^{L,S}$ is the signal strength for the long/short position respectively at the $n$-th quote. The calculation of $\omega$ is done every $\tau$ seconds;
* $\tau$ is the rebalancing period of the index;
* $r_n$ is the simple return of $U$ at the $n$-th quote:

$$
r_n = \frac{U_n - U_{n-1}}{U_{n-1}}
$$

The term $\frac{1}{1 + \omega^S_n r_n}$ does require some convexity corrections, but this is ignored as the correction is miniscule. More details in the [Appendix](#appendix).

### Indicators
Currently, only the RSI is implemented in production, but more indicators are on their way.

Note that all following lookback periods are time-based rather than tick-based. In particular:

Consider a lookback period of $T$ seconds, i.e at a received quote at tick $n$, if $t_n$ is the time we got the tick, then our lookback period is within the interval $(t_n - T, t_n]$ the number of ticks within this period is denoted as $N_n$ (For visualizations, refer to the [initial proposal](https://docs.google.com/presentation/d/1z3PncmFBdV-rygDD0mARfcXnppLAdroJnYCAbvoO0LU/edit#slide=id.g28bc25800a2_1_54) and the [implementation details](https://docs.google.com/presentation/d/1Ald5GfHc5WqIwGtqkSTLgHkCAJAYp4KpI6VK8m2LRdQ/edit#slide=id.g2d9f493af0b_0_2)).

The idea here is that the time-based window is better at capturing high-frequency changes of the underlying, as we retrieve the feed from Bloomberg, which has sub-second tick data. The more liquid the underlying, the more this method is required.

#### RSI

The [RSI](https://en.wikipedia.org/wiki/Relative_strength_index) is a technical indicator used to chart the current & historical strength/weakness of the underlying based on the closing prices of a recent trading period. It is a well-known momentum indicator, being able to detect undervaluations, overvaluations and divergences by observing the speed and change of price movements. 

The index ranges from 0 to 100 and is calculated as:

$$
RSI_n = \frac{100}{1 + RS}
$$

Where:

* $RS = \frac{SMMA(Up)_n}{SMMA(Down)_n}$ is the relative strength factor;
* $Up_n = \max(U_{n} - U_{n-1},0)$;
* $Down_n = -\min(U_{n} - U_{n-1},0)$;
* $SMMA(I)_n$ is the Wilder's smoothed moving average of an index $(I_n)_{n \geq 0}$ with weight $\frac{1}{N_n}$ determined from the time window $(t_n - T, t_n]$ at tick $n$. To initialize this average, we consider the first tick after at least the lookback period has elapsed, $n_1$:

$$
\begin{align*}
SMMA(I)_{n_1} &= SMA(\{m : t(m) \in [t(n_1) - T, t(n_1)]\},n) \\
&= \frac{1}{N_{n_1}}\sum_{j=1}^{N_{n_1}} I_j
\end{align*}
$$

Where $SMA(I, n)$ is the simple moving average of $I$ at tick $n$. At ticks $n > n_1$, it is exponentially smoothed like such:

$$
SMMA(I)_{n+1} = \frac{(N_n-1)\cdot SMMA(I)_{n} + I_{n+1}}{N_n}
$$

Note that these averages have been adapted to suit our needs as a real-time, high-frequency indicator with the modified weights used to calculate the moving averages.

##### Signal

Typically, sufficiently high RSI (exceeds an overbought level $> 50$) thresholds traders to start selling and sufficiently low RSI (less than an oversell level $< 50$) is an indicator to start buying. Here, the position taken will depend on the signal functions $\omega^L, \omega^S$, for the long/short positions respectively. Tweaking the properties of the signal radically changes the behaviour of the tactical index.

The signal functions heavily depend on a few features defining it:

###### 1. Type
There are two types of thresholds considered:

1. If the index RSI type is contrarian (RSIXAGCL/CS), $\omega^L$ is positive if $RSI \leq OS$, while $\omega^S$ is negative if $RSI \geq OB$. This corresponds to the typical contrarian strategy taken with RSI, buying when oversold and selling when overbought.
2. If the index RSI type is momentum/trend (RSIXAGML/MS), $\omega^L$ is positive if $RSI \geq OB$, while $\omega^S$ is negative if $RSI \leq OS$. This corresponds to a strategy where we ride the trend, i.e we buy when overbought and sell when oversold.

###### 2. Discrete
The signal function is discrete, that is, it is of the form of a step function, a discontinuous function, e.g for the old RSIXAGML:

$$
\omega^L(RSI) = \begin{cases} 5 & \text{if } RSI \geq 52 \\
0 & \text{if } RSI < 52 \end{cases}
$$

$$
\omega^S(RSI) = \begin{cases} 0 & \text{if } RSI > 52 \\
-1 & \text{if } RSI \leq 52 \end{cases}
$$

###### 3. Bias
The thresholds can be long or short biased. Long bias means that there is more (absolute) leverage in the long position $\omega^L$ than the short position from $\omega^S$. Short bias is the inverse.

1. The old RSIXAGCS was short biased as the short leverage is -5 while the long leverage is 1; whereas 
2. The old RSIXAGCL was long biased as its long leverage is 5, exceeding its short leverage -1.

#### MACO

The [MACO](https://en.wikipedia.org/wiki/Moving_average_crossover) is a technical indicator used to track trends using two moving averages with different degrees of smoothing. The two moving averages are of a slow (larger window) SMA and a fast (smaller window) SMA. If the fast SMA exceeds the slower SMA, this is known as a golden cross, and is used to signal bullish trends, whereas the opposite is called a death cross, signalling bearish trends. 

Unlike the usual MA Crossovers, this version is different, as traditional MACO is usually traded on evenly spaced (w.r.t trading days) candles, such as looking at daily or hourly prices. However, we intend to capture the high-frequency changes of the underlying, as it is particularly liquid, providing quotes in the span of less than a second (Bloomberg BGN/BGNE source).

Additionally, this version takes a long position when the death cross occurs instead, and a short position for the golden cross, making it "contrarian" with respect to the original.

The signal is calculated as:

$$
MACO(n) = SMA_{slow}(n) - SMA_{fast}(n)
$$

Where:

* For the $n$th tick, let $t_n$ be the time when it was quoted;
* Let $N_n$ be the number of ticks in the lookback period at tick $n$, i.e it is equal to $I_n = \#\{m : t(m) \in [t_n - T, t_n]\}$ for a lookback period $T$;
* Then $SMA_T(n)$ is the simple moving average of the underlying index over the time window $[t_n - T, t_n]$ at tick $n$:

$$
\begin{align*}
SMA_T(n) = \frac{1}{N_{n}} \sum_{U_j \in I_n} U_j 
\end{align*}
$$

Note that these averages have been adapted to suit our needs as a real-time, high-frequency indicator with the modified weights used to calculate the moving averages. Here, $SMA_{fast}$ refers to the smaller window (faster moving average) while $SMA_{slow}$ refers to the larger window (slower moving average)

##### Signal

Typically, bullish positions are taken when the fast MA exceeds the slow MA, but this version considers the opposite positions instead. This is likely an attempt to balance the index, as the typical MA Crossover is designed for profit. Here, the position taken will depend on the signal functions $\omega^L, \omega^S$, for the long/short positions respectively. Tweaking the properties of the signal radically changes the behaviour of the tactical index.

The signal functions heavily depend on a few features defining it:

###### 1. Type
There is only one type of threshold considered:

1. When $MA_{slow} > MA_{fast}$, go long, otherwise go short. 

###### 2. Discrete
The signal function is discrete, that is, it is of the form of a step function, a discontinuous function, e.g for MACOXAG1:

$$
\omega^L(MACO) = \begin{cases} 2 & \text{if } MACO > 0 \\
0 & \text{if } MACO \leq 0 \end{cases}
$$

$$
\omega^S(MACO) = \begin{cases} 0 & \text{if } MACO \geq 0 \\
-1 & \text{if } MACO < 0 \end{cases}
$$

###### 3. Bias
The thresholds can be long or short biased. Long bias means that there is more (absolute) leverage in the long position $\omega^L$ than the short position from $\omega^S$. Short bias is the inverse.

1. The MACOUSDJPY1 was short biased as the short leverage is -4 while the long leverage is 2; whereas 
2. The MACOXAG1 was long biased as its long leverage is 2, exceeding its short leverage -1.

### General Parameters & Considerations

#### Transition State
This applies to the RSI, Bollinger Bands and Pair Trading. Let $upper$ and $lower$ correspond to the upper and lower thresholds respectively:

Typically, if there is a non-zero gap between the thresholds, i.e if $upper > lower$, there will be a neutral region $(lower,upper)$ between the two thresholds. There are two choices for what we can do in this region:

1. If the transition state is `long cash`, we just take a long cash/zeroed position. The signal functions would look like the following for the contrarian case. This is just a pair of step functions:

$$
\omega^L(SIG) = \begin{cases} 0 & \text{if } SIG \geq uplowerper \\
L_l & \text{if } SIG < lower \end{cases}
$$

$$
\omega^S(SIG) = \begin{cases} L_u & \text{if } SIG > upper \\
0 & \text{if } SIG \leq upper \end{cases}
$$

2. If the transition state is `hold`, we do not zero the position, but preserve the last non-zero position. e.g, for the contrarian case, if the last threshold touched was the upper level, $\omega^L$ will be positive and $\omega^S$ will be 0 until the moment that the RSI hits the lower level, which then $\omega^L$ will be 0 and $\omega^S$ will be negative. In the contrarian case, the signal functions look like:

$$
\omega^L(SIG) = \begin{cases}
    L_l & \text{if }lower \text{ was the last level touched} \\
    0 & \text{otherwise}
\end{cases}
$$

$$
\omega^S(SIG) = \begin{cases}
    L_u & \text{if }upper \text{ was the last level touched} \\
    0 & \text{otherwise}
\end{cases}
$$

This second transition state was considered as 

1. The number of signal changes per day are heavily reduced.
2. There are not too many flat periods

Note that if $upper = lower$ there can be issues with the signals changing too fast and forcing the index to continuously rise/fall. This was seen with early iterations of RSIXAGML.


#### Market Hours
The tactical index can only run when the underlying market has opened. When the first tick in a day comes in, there will be a period of time where no points will be generated, equal to the (largest) lookback time. More details in the [spreadsheet](https://docs.google.com/spreadsheets/d/1l75Bepr-1zcOuifR8bvZhz0ofNxtfAGVZIFU1drp0FM/edit?gid=1546912591#gid=1546912591). & the section below



#### Lookback
When the market opens (the first received quote) at time $t_{n_0}$, the lookback period begins then and ends at the first tick after $t_{n_0} + T$, say $t_{n_1}$. The decision to not generate tactical ticks in this lookback period was made as:

1. The underlyings are usually very volatile within the first hour of market opening; and
2. A small number of points will make the SMMAs much more volatile.

However, this means that we only start generating the index, up to 1-2 hours after the underlying has started, losing a lot of potential trading time. Do note that Cryptocurrencies do not face this issue, but the implementation is kept same for convenience.

To partially mitigate this issue, there is an opening lookback time $T' < T$ such that at the first tick after time $t_{n_0} + T'$, say $t_{n_{0.5}}$, the Tactical starts generating points with lookback = $T'$ UNTIL $t_{n_1}$, where the index switches to the lookback of $T$. Refer to [slides 3-4](https://docs.google.com/presentation/d/1teH_L1ZwjqzE_See0uYQIHxQyZKCAEkeLqPGqvtLBtc/edit#slide=id.g28bc25800a2_1_54) for the original proposal and the [implementation details](https://docs.google.com/presentation/d/1Ald5GfHc5WqIwGtqkSTLgHkCAJAYp4KpI6VK8m2LRdQ/edit#slide=id.g2d9f493af0b_0_2).

#### Rebalancing

To avoid unecessary volatility from bid/ask bounce, the rebalancing is done every $\tau$ seconds instead of at the same time as the tick generation. E.g, if $\tau = 1$, then rebalancing is done every 1s from the start of the lookback period (rounded to the nearest second). Note that effectively, this calculation is done when the next tick is generated. Refer to [slides 3-5](https://docs.google.com/presentation/d/1Ald5GfHc5WqIwGtqkSTLgHkCAJAYp4KpI6VK8m2LRdQ/edit#slide=id.g2c51ed7aaf2_0_113) for examples.

![](../images/rebalancing.png)

#### Spread

The current choice is to leave the spread entirely up to the Dealers. As such, the below is implemented, but $\alpha$ and $commission$ is 0.

_OLD_:

There were 4 choices for the spread ([link](https://github.com/clement-deriv/quants/blob/master/backend_product_specification/Tactical%20index/Research/Tactical%20index%20-%20Spread%20proposal.md)), including a [probabilistic spread](https://docs.google.com/document/d/1QhMbhAaK0narCv2CXX-6nZfHaXVvRthuVOB-3wI0Sf0/edit) (Validated [here](https://docs.google.com/document/d/1bQJD9Z1U6_ynxUMlyE1ZwWcVBj9yptb7UD377tn1t7c/edit)). 

The final decision was to consider a spread of:

$$
Spread = \alpha \cdot calibration + commission
$$

Where
* $\alpha$ is some number, usually 1;
* $calibration$ is dependent on quant calculations;
* $commission$ is dealer-set.

All these terms are in absolute values and can be set at any time through BO.

### Links & Scope
[Slack Canvas](https://deriv-group.slack.com/docs/T0D277EE5/F061TAC21Q9) 

## Appendix
### Deriving the drift correction term
Here, we want our simple returns to be 1, instead of our log returns. We expand into Taylor's series:

$$
\begin{align}
\mathbb{E}\left[\frac{S_{t+1}}{S_t}\right] &= 1 \nonumber\\
1 &= \mathbb{E}\left[\frac{1+\omega_L r_L}{1+\omega_S r_S}e^{\textrm{correction}_{\textrm{drift}}}\right] \nonumber\\
e^{-\textrm{correction}_{\textrm{drift}}} &= \mathbb{E}\left[\frac{1+\omega_L r_L}{1+\omega_S r_S}\right] \nonumber\\
1 - \textrm{correction}_{\textrm{drift}} + \cdots &= \mathbb{E}[(1+\omega_L r_L)(1-\omega_Sr_S + \omega_S^2 r_S^2 + \cdots)] \nonumber\\
1 - \textrm{correction}_{\textrm{drift}} &\approx 1+\omega_L \mathbb{E}[r_L] - \omega_S \mathbb{E}[r_S] - \omega_L \omega_S \mathbb{E}[r_L r_S] + \omega_S^2 \mathbb{E}[r_S^2]
\end{align}
$$

With the underlying assumption:

$$
|\omega_Sr_S| \ll 1
$$


We now consider the relationship between the log returns & simple returns of the underlying:

$$
\mathbb{E}\left[\ln\left(\frac{U_{t+1}}{U_t}\right)\right] = \mathbb{E}[\ln(1+r)] \approx \mathbb{E}[r] - \frac{\mathbb{E}[r^2]}{2}
$$

Recalling the initial assumption that the underlying is modeled like a driftless GBM, we see that:

$$
- \frac{\sigma^2dt}{2} = \mathbb{E}\left[\ln\left(\frac{U_{t+1}}{U_t}\right)\right] \approx \mathbb{E}[r] - \frac{\mathbb{E}[r^2]}{2}
$$


Therefore $r \approx \sigma\sqrt{dt} \, dB_t$, we see that $\mathbb{E}[r] \approx 0$. 

With this, comes the assumption

$$
|\omega_Lr_S| \ll 1
$$

Applying this to equation (1), we see that:

$$
\textrm{correction}_{\textrm{drift}} \approx \omega_L \omega_S \mathbb{E}[r_L r_S] - \omega_S^2 \sigma_S^2 dt
$$

Do note that the expansion $\log(1+x) = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n}$ has radius of convergence = 1, which implies that if $|\omega r| \geq 1$, the logic breaks (That is, if it already hasn't been broken by the small-value approximation being invalidated)

Furthermore, we see that $\omega^L \omega^S = 0$ as they can not be non-zero at the same time, therefore the correlation term drops out of the correction term:

$$
\textrm{correction}_{\textrm{drift}} \approx - \omega_S^2 \sigma_S^2 dt
$$

As we are working with terms up to the second order, we can relax the assumptions to the following:

$$
\begin{gather*}
\mathbb{E}[r_S] = \mathbb{E}[r_L] = 0 \tag{Driftless GBM} \\
\omega_L \sigma_L\sqrt{dt} \ll 1 \\
\omega_S \sigma_S\sqrt{dt} \ll 1
\end{gather*}
$$

Since the ticks are discretely sampled, with uneven time-spacing, the second and third requirements are clearly only valid for very small $dt$.

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.