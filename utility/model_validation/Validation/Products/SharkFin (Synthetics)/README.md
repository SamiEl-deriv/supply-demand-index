<!---
$$\beta=-\frac{\zeta(\frac{1}{2})}{\sqrt{2\pi}}\approx0.5826$$ 

$$
\begin{aligned}
\alpha =& e^{\phi\times \beta\times\sigma\times\sqrt{dt}} \\
=& \left(e^{phi\times \beta\times\sqrt{dt}}\right)^{\sigma}\\
\end{aligned}
$$

where, 
* $\phi=1 \text{ for call and } -1 \text{ for put}$
* $dt = \sqrt{\frac{1}{365\times86400}}$
* $\beta=-\frac{\zeta(\frac{1}{2})}{\sqrt{2\pi}}\approx0.5826$

$$
\begin{aligned}
\text{For Call:}\\
\alpha =& 1.0078046960172629^{sigma} \\
\text{For Put:}\\
\alpha =& 0.992255745534719^{sigma} \\
\end{aligned}
$$


$$U=\text{Barrier}\times\alpha$$



$$\theta=\frac{\mu}{\sigma}-0.5\sigma$$ 

$$v= \sqrt{\max\left(\theta^2+2\times r_q,0\right)}$$ 

$$\phi = 1 \text{ if } \text{Barrier}>S \text{ else} -1$$ 

$$e_1= \frac{ln(\frac{S}{U})-\sigma\times v\times t}{\sigma\sqrt{t}}$$ 

$$e_2= \frac{-ln(\frac{S}{U})-\sigma\times v\times t}{\sigma\sqrt{t}}$$ 

Notes: <br> 1. Volatility is pre-determined [here](https://github.com/regentmarkets/bom-market/blob/master/config/files/flat_volatility.yml). <br> 2. $v$ is calculated such as the term is always positive when $r_q$ is too negative. <br> 3. $\mu$ is zero for the volatility index. <br> 4. [This document](http://www.matthiasthul.com/wordpress/wp-content/uploads/2015/06/AmericanDigitalCall.pdf) provides detailed explanation on how to derive the one touch formula. <br> 5. $\beta$ is a correction term for barrier option that improve the approximation of a discrete-time prices using continuous-time formulas. The correction term shifts the expected maximum or minimum price. [More information here.](https://www0.gsb.columbia.edu/mygsb/faculty/research/pubfiles/1851/lookback2.pdf) <br>  6. $\zeta$ is the Riemann zeta function.|[1. perl-Pricing-Engine-BlackScholes](https://github.com/regentmarkets/perl-Pricing-Engine-BlackScholes) <br> [2. perl-Math-Business-BlackScholesMerton](https://github.com/binary-com/perl-Math-Business-BlackScholesMerton)<br> [3. BarrierBuilder.pm](https://github.com/regentmarkets/bom/blob/master/lib/BOM/Product/Role/BarrierBuilder.pm)|


--->
# SharkFin on Volatility Indexes Validation Report
#### Updated 17/01/2024

### <u>Squad Members</u>
|Area | Person In Charge | 
| :- | :- | 
| Squad Leader | - | 
| Project Manager | Ekaterina (Kate) (ekaterina@deriv.com) | 
| Product Owner | Kai Jie (kaijie@deriv.com) | 
| Backend | - | 
| Frontend | - | 
| QA | - | 
| Model Validation | Matthew (matthew.chan@deriv.com) |

## Summary
<div align='justify'>
The validation process for SharkFin options on synthetics covers the product specifications, Dtrader, API testing and validating the calculations including offered barriers (Intraday and Daily+), payout per point, contract value, sold payout and expired payout calculations. Contract parameters such as call/put, underlying and duration were randomly selected from all volatility indexes and all duration options. Vol Markup was set at 0.025, values for Spread Spot and Black Scholes Markups were also added randomly in back office to verify their calculations. Testing on risk monitoring functions available in BO were also conducted. There are a total of 4 observations found, of which there are no blockers, 4 non-blockers and 0 resolved observations.<br>

Report in the form of slides [here](https://drive.google.com/drive/folders/18oqwDVaKsg2uJlNerSbsAajxQWBPUHzG?usp=sharing)
 
</div>

## Product Information
#### <u>Product Description</u>
<div align='justify'>

Perl code version: - <br><br>
This product allows the client to express a moderate bullish or bearish view on an underlying asset.

The buyer of the SharkFin Call/Put will participate in any increase/decrease above/below a predetermined strike. However, the client bears the risk that the SharkFin option will be knocked out and early terminated if the asset spot price breaches a predetermined barrier depending on the contract type. In this case, the client will receive a rebate equal to a multiple of its initial stake.

There will be 2 types of SharkFins, the first one is SharkFin Any Time KO, where it is a path-dependent contract (contract gets knocked out if the spot touches/crosses the barrier during the contract lifetime) and the second one is SharkFin at Expiry KO, where it is not a path-dependent contract (contract gets knocked out if the spot touches/crosses the barrier at maturity).

Since the SharkFin Call/Put Any Time KO option is knocked out above/below the barrier, this option is offered when the asset spot price is below/above this barrier at the start of the trade. Furthermore, clients can further select a barrier that equals the strike in SharkFin at Expiry KO options.
</div>

#### <u>Term & Conditions</u>
Client Pays: 
$$Stake$$

Client Receives:<br>
<ins> **Any Time KO** </ins>

| **SharkFins**  | **If barrier has not ever been touched/crossed, clients receive** | **If barrier has ever been touched/crossed, clients receive** | 
| :---:  | :---: | :---: |  
| **SF Call**  | $$\text{Payout Per Point} \times (\text{Final Price} - \text{Strike Price})$$ | $$\alpha \times \text{Stake}$$ | 
| **SF Put** | $$\text{Payout Per Point} \times (\text{Strike Price} - \text{Final Price})$$ | $$\alpha \times \text{Stake}$$ | 

<ins> **At Expiry KO** </ins>

| **SharkFins**   | **If barrier is not touched/crossed at expiry, clients receive** | **If barrier is touched/crossed at expiry, clients receive** | 
| :---: | :---: | :---: |  
| **SF Call**  | $$\text{Payout Per Point} \times (\text{Final Price} - \text{Strike Price})$$ | $$\alpha \times \text{Stake}$$ | 
| **SF Put** | $$\text{Payout Per Point} \times (\text{Strike Price} - \text{Final Price})$$ | $$\alpha \times \text{Stake}$$ | 

Where,<br>
$\text{Payout Per Point}_{Call|Put} = \frac{{Stake} \times (1 - \alpha {KO}^{Ask}(call|put))}{{UOC}^{Ask}|{DOP}^{Ask}}$ <br>
$\alpha= \text{Rebate Percentage}$

### Product Pricing
Two types of SharkFins are offered: Anytime KO; at Expiry KO. The product can be divided into two parts: Part with rebate; Part without rebate. 

The part with rebate will be priced with a digital option multiplied by the stake and rebate percentage of the stake. While theremaining stake will go into the part without rebate which is priced with an up-and-out call / down-and-out put. For Any Time KO SharkFins, American verions of the options will be used while European versions of the options will be used for at Expiry KO SharkFins. 

<!---

$$\begin{align*}
\text{Stake SharkFin}\^{\text{Any Time/Expiry}}\_{Call|Put} &= \text{Part with rebate} + \text{Part without rebate} \\
&=\alpha\times\text{Digital}\times\text{Stake}+\text{Stake}_{UOC|DOP}
\end{align*}$$

#### SharkFin at Any Time KO
##### Part with rebate:
American Single Barrier Digital Option aka One Touch is used to price the part with rebate.

$$
\begin{aligned}
KO(S_t, B, r, T - t, \sigma, call, any\ time) &= 1_{\left(\underset{0 \leq t \leq T}\max{S_t}\right) < B}\Bigg \lbrace N(e_+)\left(\frac{B}{S_t}\right)^{\frac{\theta_- + \vartheta_-}{\sigma}} + N(-e_-)\left(\frac{B}{S_t}\right)^{\frac{\theta_- - \vartheta_-}{\sigma}}\Bigg \rbrace \\
KO(S_t, B, r, T - t, \sigma, put, any\ time) &= 1_{\left(\underset{0 \leq t \leq T}\min{S_t}\right) > B}\Bigg \lbrace N(-e_+)\left(\frac{B}{S_t}\right)^{\frac{\theta_- + \vartheta_-}{\sigma}} + N(e_-)\left(\frac{B}{S_t}\right)^{\frac{\theta_- - \vartheta_-}{\sigma}}\Bigg \rbrace
\end{aligned}
$$

where, 
* $S_t$: Spot price
* $B$: Barrier
* $r$: Discount rate
* $T - t$: Time remaining to maturity
* $\sigma$: Volatility
* $e_\pm = \frac{\pm \ln \left(\frac{S_t}{B}\right) - \sigma \vartheta_-(T - t)}{\sigma \sqrt{T - t}}$
* $\theta_- = -\frac{\sigma}{2}$
* $\vartheta_- = \sqrt{\theta^2_- + 2r}$

##### Part without rebate:
Up-and-out call (for SharkFin Call) or Down-and-out Put (for SharkFin Put) knock-out options are used to price the part without rebate of SharkFins.

$$
\begin{aligned}
UOC(S_t, K, B, r, T - t, \sigma, any\ time) &= 1_{\left(\underset{0 \leq t \leq T} \max{S_t}\right) < B} \bigg \lbrace C_{v}(S_t, K, r, T - t, \sigma) - C_{v}(S_t, B, r, T - t, \sigma) - (B - K)C_{d}(S_t, B, r, T - t, \sigma) \\
& \quad-\frac{S_t}{B}\left[C_{v}\left(\frac{B^2}{S_t}, K, r, T - t, \sigma \right) - C_{v}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right)- (B - K)C_{d}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right)\right] \bigg \rbrace \\
DOP(S_t, K, B, r, T - t, \sigma, any\ time) &= 1_{\left(\underset{{0 \leq t \leq T}}\min{S_t}\right) > B}\bigg \lbrace P_{v}(S_t, K, r, T - t, \sigma) - P_{v}(S_t, B, r, T - t, \sigma) - (K - B)P_{d}(S_t, B, r, T - t, \sigma) \\
& \quad-\frac{S_t}{B}\left[P_{v}\left(\frac{B^2}{S_t}, K, r, T - t, \sigma \right) - P_{v}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right) - (K - B)P_{d}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right)\right]\bigg \rbrace
\end{aligned}
$$

where,
* $C_{v}(S_t, K, r, T - t, \sigma) = [N(d_1)S_t - N(d_2)K]e^{-r(T - t)}$
* $C_{d}(S_t, K, r, T - t, \sigma) = N(d_2)e^{-r(T - t)}$
* $P_{v}(S_t, K, r, T - t, \sigma) = [N(-d_2)K - N(-d_1)S_t]e^{-r(T - t)}$
* $P_{d}(S_t, K, r, T - t, \sigma) = N(-d_2)e^{-r(T - t)}$
* $d_1 = \frac{\ln \left(\frac{S_t}{K}\right) + \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$
* $d_2 = \frac{\ln \left(\frac{S_t}{K}\right) - \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$

#### SharkFin at Expiry KO
##### Part with rebate:

$$
\begin{aligned}
KO(S_t, B, r, T - t, \sigma, call, expiry) &= C_d(S_t, B, r, T - t, \sigma) \\
KO(S_t, B, r, T - t, \sigma, put, expiry) &= P_d(S_t, B, r, T - t, \sigma)
\end{aligned}
$$

where,
* $d_2 = \frac{\ln \left(\frac{S_t}{K}\right) - \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$
* $C_{d}(S_t, K, r, T - t, \sigma) = N(d_2)e^{-r(T - t)}$
* $P_{d}(S_t, K, r, T - t, \sigma) = N(-d_2)e^{-r(T - t)}$
* $S_t$: Spot price
* $B$: Barrier
* $r$: Discount rate
* $T - t$: Time remaining to maturity
* $\sigma$: Volatility
  
##### Part without rebate:

$$
UOC(S_t, K, B, r, T - t, \sigma, expiry) = C_v(S_t, K, r, T - t, \sigma) - C_v(S_t, B, r, T - t, \sigma) - (B -K) C_d(S_t, B, r, T - t, \sigma)\\
DOP(S_t, K, B, r, T - t, \sigma, expiry) = P_v(S_t, K, r, T - t, \sigma) - P_v(S_t, B, r, T - t, \sigma) - (K -B) P_d(S_t, B, r, T - t, \sigma)
$$

where,
* $C_{v}(S_t, K, r, T - t, \sigma) = [N(d_1)S_t - N(d_2)K]e^{-r(T - t)}$
* $C_{d}(S_t, K, r, T - t, \sigma) = N(d_2)e^{-r(T - t)}$
* $P_{v}(S_t, K, r, T - t, \sigma) = [N(-d_2)K - N(-d_1)S_t]e^{-r(T - t)}$
* $P_{d}(S_t, K, r, T - t, \sigma) = N(-d_2)e^{-r(T - t)}$
--->

#### Offering

Instrument:
* Volatility Index 10/25/50/75/100/150/250 (1s)
* Volatility Index 10/25/50/75/100 (2s)

Duration:
* 1 minute to 1440 minutes
* 1 hour to 24 hours
* 1 day to 365 days

#### Bid-Ask Spread
<!---
There are additional charges on top of the theoretical Black-Scholes pricing formula in the form of Delta Charge, Volatility Charge and Additional Markup.

Volatility charge is added directly into the Black-Scholes formula by adjusting the value of sigma as below to calculate $BS_{mid}$

$$
\begin{aligned}
\sigma_{ask} &= \sigma_{mid}\times(1+vol\space markup) \\
\sigma_{bid} &= \sigma_{mid}\times(1-vol\space markup) \\
\end{aligned}
$$

Then, delta charge and additional markup are added directly on top.

$$
\begin{aligned}
BS_{ask} &= BS_{mid}+|\Delta|\times\frac{Spot\space Spread}{2}+BS_{markup} \\
BS_{bid} &= BS_{mid}-|\Delta|\times\frac{Spot\space Spread}{2}-BS_{markup} \\
\end{aligned}
$$

Spot spread and $\text{BS}_{\text{markup}}$ are set in Back Office. Ask price is applicable when buying any vanilla contract. Bid price is only applicable for selling a contract before its maturity.
--->

## Validation Methodology
### Overview
<!---
<div align='justify'>
The product specifications were validated focussing on pricing formulas and commission structure for ask/bid price was validated. 
 
All indexes that offer Vanilla Options were validated along with all time offerings. The indexes along with contract durations and barriers/strikes were selected randomly (max and min durations were also selected) when purchasing contracts from the API. Markup values were randomly added in Back Office during testing. BO Parameters checked included:

1) Vol Markup 
2) Spread Spot	
3) Delta Config 
4) Black Scholes Markup	
5) Min Number Of Contracts
6) Max Open Position	 
7) Max Daily Volume	
8) Max Daily PnL 
9) Global Potential Loss	
10) Global Realised Loss	
 
 </div>
--->
### Assumptions

<!---
All fiat currencies (USD, EUR, GBP, AUD) were tested however for cryptocurrencies, only BTC and ETH were tested as it was assumed that all cryptocurrencies will perform identically.

Most testing was done in the QA Box environment as many fixes were made ad hoc with BE team and deployed straight into our QA Box for testing. 

Testing for Back Office were done with the assumptions that reasonable values are keyed in. For example, "vol markup" should never be a negative value or that "Delta Config" should never be more than 1 or less than 0. 
--->

### Validation Areas
<!---
1) Validate Barrier calculations
2) Validate Payout per Point
3) Validate Open Contracts' value (Current Bid Price)
4) Validate Expired Contracts' Payout
5) Validate Manually Sold Contracts Payout
6) Validate Implementation of Markups (Delta Charge and BS Markup)
7) Validate DTrader (Desktop and Mobile)
8) Validate Back Office functions
--->

## Validation Results
### Results Statistic
| Scope | Presentation Slides | # Observations | Resolved | Non-blocker | Blocker |
| :- | :-: | :-: | :-: | :-:|:-:|
| <b>Total Observations </b>| -    | 4 | - | 4 | - |
| Specification             | Link | 1  | -  | 1 | - |
| Backend Code              | Link | 3  | - | 3 | - |
| DTrader                   | Link | - | - | - | - |
| API                       | Link | - | - | - | - |
| DTrader (Mobile)          | Link | - | - | - | - |
| Back Office               | Link | - | - | -  | - |

    
### Specifications
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Ask Price for Shark Fin at Expiry KO could be less than Mid Price if Spot Charge commission is set above a certain threshold.| Non-blocker |

### Backend Code    
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Payout formula incorrect, should include: stake * alpha  | Non-blocker |
|2 | Incorrect formula for ask price for anytime sharkfin KO (Put). | Non-blocker |
|3 | Cd and Pd for “rebate part” of SharkFins only defined in “without Rebate part” of specs. | Non-blocker |


<!---

$$
\begin{array}{ll}
{UOC}^{Ask} &= UOC(S_0, B(1 + \Delta_B), \sigma) + |UOC(S_0(1 + \Delta_S), B(1 + \Delta_B), \sigma) - UOC(S_0, B(1 + \Delta_B), \sigma)| \\
&\quad+ |UOC(S_0, B(1 + \Delta_B), \sigma(1 + \Delta_\sigma)) - UOC(S_0, B(1 + \Delta_B), \sigma)| + c_1 \\
{DOP}^{Ask} &= DOP(S_0, B(1 - \Delta_B), \sigma) + |DOP(S_0(1 - \Delta_S), B(1 - \Delta_B), \sigma) - DOP(S_0, B(1 - \Delta_B), \sigma)| \\
&\quad+ |DOP(S_0, B(1 - \Delta_B), \sigma(1 + \Delta_\sigma)) - DOP(S_0, B(1 - \Delta_B), \sigma)| + c_1 \\
{KO}^{Ask}(call) &= KO(S_0(1 + \Delta_S), B(1 - \Delta_B), \sigma(1 + \Delta_\sigma), call) + c_2 \\
{KO}^{Ask}(put) &= KO(S_0(1 - \Delta_S), B(1 + \Delta_B), \sigma(1 + \Delta_\sigma), put) + c_2 
\end{array}
$$

MV:<br>
Up and out call 

$$
\begin{align}
c_{uo} =&\space c - c_{ui} \\
=&\space SN(d_1)e^{-qt} - Ke^{-rt}N(d_2) - \biggl[ SN(x_1)e^{-qt} - Ke^{-rt}N(x_1-\sigma\sqrt{t})- \\ 
&\space Se^{-qt} \left( \frac{B}{S} \right) ^{2\lambda} \left[ N(-y)-N(-y_1) \right] + Ke^{-rt} \left(\frac{B}{S} \right) ^{2\lambda-2} \
\left[ N(-y+\sigma\sqrt{t})-N(-y_1+\sigma\sqrt{t}) \right] \biggr]
\end{align}
$$

where,<br>

$$
\begin{align}
r &= 0\\
q &= 0\\
\lambda &= \frac{r-q+\sigma^2/2}{\sigma^2}=\frac{1}{2}\\
d_1 &= \frac{\ln{\frac{S}{K}}+(r-q+\sigma^2/2)t}{\sigma\sqrt{t}}=\frac{\ln{\frac{S}{K}}+(\sigma^2/2)t}{\sigma\sqrt{t}}\\
d_2 &= d_1-\sigma\sqrt{t}\\
y &= \frac{\ln{\frac{B^2}{SK}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t}= \frac{\ln{\frac{B^2}{SK}}}{\sigma\sqrt{t}}+\frac{1}{2}\sigma\sqrt{t}\\
x_1 &= \frac{\ln{\frac{S}{B}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t}= \frac{\ln{\frac{S}{B}}}{\sigma\sqrt{t}}+\frac{1}{2}\sigma\sqrt{t}\\
y_1 &= \frac{\ln{\frac{B}{S}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t} = \frac{\ln{\frac{B}{S}}}{\sigma\sqrt{t}}+\frac{1}{2}\sigma\sqrt{t}\\
\end{align}
$$

<br>then: 

$$
\begin{align}
c_{uo} =&\space SN(d_1) - KN(d_2) - SN(x_1) + KN(x_1-\sigma\sqrt{t}) \\
&+BN(-y) - BN(-y_1) - K\frac{S}{B}N(-y+\sigma\sqrt{t}) + K\frac{S}{B}N(-y_1+\sigma\sqrt{t})\\
=&\space SN(d_1) - KN(d_2) - SN(x_1) + KN(x_1-\sigma\sqrt{t}) \\
&+B(1-N(y))- B(1-N(y_1)) - K\frac{S}{B}(1-N(y-\sigma\sqrt{t})) + K\frac{S}{B}(1-N(y_1-\sigma\sqrt{t}))\\
=&\space SN(d_1) - KN(d_2) - SN(x_1) + KN(x_1-\sigma\sqrt{t}) \\
&-BN(y) +BN(y_1) + K \frac{S}{B} N(y-\sigma\sqrt{t}) - K\frac{S}{B}N(y_1-\sigma\sqrt{t})\\
\end{align}
$$

Sharkfin Spec UOC:

$$
\begin{align}
UOC(S_t, K, B, r, T - t, \sigma, any\ time) =&\space  C_{v}(S_t, K, r, T - t, \sigma) - C_{v}(S_t, B, r, T - t, \sigma) - (B - K)C_{d}(S_t, B, r, T - t, \sigma) \\
& \space-\frac{S_t}{B}\left[C_{v}\left(\frac{B^2}{S_t}, K, r, T - t, \sigma \right) - C_{v}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right)- (B - K)C_{d}\left(\frac{B^2}{S_t}, B, r, T - t, \sigma \right)\right]  \\
\end{align}
$$

where,
* $C_{v}(S_t, K, r, T - t, \sigma) = [N(d_1)S_t - N(d_2)K]e^{-r(T - t)}$
* $C_{d}(S_t, K, r, T - t, \sigma) = N(d_2)e^{-r(T - t)}$

and
* $d_1 = \frac{\ln \left(\frac{S_t}{K}\right) + \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$
* $d_2 = \frac{\ln \left(\frac{S_t}{K}\right) - \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$
then:<br>

$$
\begin{aligned}
UOC(S_t, K, B, r, T - t, \sigma, any\ time) =&\space SN(d_1)-KN(d_2)-SN(x1)+BN(x_1-\sigma\sqrt{t})-B N(x_1-\sigma\sqrt{t}) +KN(x_1-\sigma\sqrt{t}) \\
&-\frac{S}{B} \left(\frac{B^2}{S}N(y)\right) + \frac{S}{B} \left(KN(y-\sigma\sqrt{t})\right) +\frac{S}{B} \left(\frac{B^2}{S}N(y_1)\right) - \frac{S}{B} \left(BN(y_1-\sigma\sqrt{t})\right) \\
&+\frac{S}{B}BN(y_1-\sigma\sqrt{t}) - \frac{S}{B}KN(y_1-\sigma\sqrt{t})\\
=&\space SN(d_1)-KN(d_2)-SN(x1)+BN(x_1-\sigma\sqrt{t})-B N(x_1-\sigma\sqrt{t}) +KN(x_1-\sigma\sqrt{t})\\
&-BN(y) + \frac{S}{B} \left(KN(y-\sigma\sqrt{t})\right) +BN(y_1) - SN(y_1-\sigma\sqrt{t}) \\
&+SN(y_1-\sigma\sqrt{t}) - \frac{S}{B}KN(y_1-\sigma\sqrt{t})\\
=&\space SN(d_1)-KN(d_2)-SN(x1) +KN(x_1-\sigma\sqrt{t}) \\
&-BN(y) +BN(y_1) + K \frac{S}{B} N(y-\sigma\sqrt{t}) - K\frac{S}{B}N(y_1-\sigma\sqrt{t})\\
\end{aligned}
$$
--->

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.