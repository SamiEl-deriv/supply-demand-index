# Vanilla Options on Volatility Indexes Validation Report 
#### Updated 28/08/2023 

### <u>Squad Members</u>
|Area | Person In Charge | 
| :- | :- | 
| Squad Leader | Magdalena (magdalena.nitefor@deriv.com) | 
| Project Manager | Ekaterina (Kate) (ekaterina@deriv.com) | 
| Product Owner | Alassane (alassane@deriv.com) | 
| Backend | Qin Feng (qinfeng@deriv.com) | 
| Frontend | Akmal (akmal@deriv.com) | 
| QA | Jishi (jishi@deriv.com) / Luqman (luqman@deriv.com) | 
| Model Validation | Matthew (matthew.chan@deriv.com) |

## Summary
<div align='justify'>
The validation process for vanilla options on synthetics covers Dtrader (Desktop and Mobile), API testing and validating the calculations including offered barriers (Intraday and Daily+), payout per point, contract value, sold payout and expired payout calculations. Contract parameters such as call/put, underlying and duration were randomly selected from all volatility indexes and all duration options. Vol Markup was set at 0.025, values for Spread Spot and Black Scholes Markups were also added randomly in back office to verify their calculations. Testing on risk monitoring functions available in BO were also conducted. There are a total of 43 observations found, of which there are no blockers, 34 non-blockers and 9 resolved observations.<br>

Report in the form of slides [here](https://drive.google.com/drive/folders/1WsfWmymVj_PP_3WLMf5r4L806J1ojGok?usp=share_link)
 
</div>

## Product Information

#### <u>Product Description</u>
Perl code version: https://redmine.deriv.cloud/issues/95135#Enable_vanilla_synthetics_on_real <br><br>
Vanilla options allow you to express a bullish or bearish view on an index by purchasing a "<b>Call</b>" or a "<b>Put</b>".

If you buy a "<b>Call</b>" option, you receive a Payout at Expiry if the Final Price is above the Strike Price. Otherwise, your "<b>Call</b>" option will expire worthless.

If you buy a "<b>Put</b>" option, you receive a Payout at Expiry if the Final Price is below the Strike Price. Otherwise, your "<b>Put</b>" option will expire worthless.

The Payout is equal to the Payout Per Point multiplied by the difference between the Final Price and the Strike Price. You will make profit if your Payout is higher than your initial stake.

If you choose to sell back your option before the Expiry, you receive a Contract Value which depends on the prevailing spot price and the remaining time to expiry.
#### <u>Term & Conditions</u>

Client Pays: 

$$Stake$$

Client Receives:<br>
$\quad$ For Call:
   
$$\begin{align*}
Payout = Payout\space per\space Point\times\max({S_t-K,0}) 
\end{align*}$$

$\quad$ For Put:
   
$$\begin{align*}
Payout = Payout\space per\space Point\times\max({K-S_t,0}) 
\end{align*}$$

where

* $\text{Payout Per Point} = \frac{\text{Stake}}{\text{Option}_{Ask}}$

#### <u>Barrier</u>

Before the barriers are calculated, the spot is first rounded based on the below formula, the rounding is to stabilise the offered barriers.

$$\begin{align*}
\text{Expected Move (1 min)}&=\text{Spot}\times \text{vol}\times\sqrt{\frac{60}{365\times86400}}\\
\text{Number of Digits}&=\text{Round}\left(\log_{10}\left(\frac{1}{ \text{Expected Move (1 min)}}\right),\space0\right)-1\\
\text{Spot} &= \text{Round}(\text{Spot}, \text{ Number of Digits})
\end{align*}$$

The barriers/strikes are calculated from this rounded spot.<br>

#### Intraday Barriers:

For Intraday Strikes, the Strikes are to how far ITM or OTM the option can be with respect to Delta, For a corresponding delta, the strike, $K$ will be:

$$\begin{align*}
K = S_t e^{\frac{\sigma^2}{2}(T)-\sigma\sqrt{T}\times N^{-1}(\Delta)} \text{ for }\Delta=0.1,\space0.3,\space0.5,\space0.7,\space0.9\\\\
\end{align*}$$

The Strikes are then rounded to the 

number of decimals as the corresponding instrument. The configuration for Deltas can also be set in Back Office


#### Daily+ Barriers:
For Daily+ Barriers the central strike $(K_{central})$, max strike $(K_{max})$, and min strike $(K_{min})$ are first calculated, then the strike steps are calculated based on the maximum allowed number of strikes $(N)$ set in Back Office. $K_{max}$ corresponds to $\Delta_{min}$ and $K_{min}$ corresponds to $\Delta_{max}$

$$\begin{align*}
K_{central} &= \text{Round}(\text{Spot}, \text{ Number of Digits})\\
K_{max} &= \text{Round Down}\left(S_t e^{\frac{\sigma^2}{2}(T)-\sigma\sqrt{T}\times N^{-1}(\Delta)},\text{ Number of Digits} \right), \Delta = 0.1\\
K_{min} &= \text{Round Up}\left(S_t e^{\frac{\sigma^2}{2}(T)-\sigma\sqrt{T}\times N^{-1}(\Delta)},\text{ Number of Digits} \right), \Delta = 0.9\\\\
\end{align*}$$

Where,<br>

$\text{Strike Step}_1 = \frac{K_m{}_a{}_x-K_c{}_e{}_n{}_t{}_r{}_a{}_l}{\frac{N-3}{2}+1} $<br>

$\text{Strike Step}_2 = \frac{K_c{}_e{}_n{}_t{}_r{}_a{}_l-K_m{}_i{}_n}{\frac{N-3}{2}+1}$

Strikes in between $K_{min}$ and $K_{central}$ are then $K_{min}\times i \times \text{Strike Step}_1$ for $i$ ranging from 1 to $\frac{N-3}{2}$. 

Strikes in between $K_{central}$ and $K_{max}$ are then $K_{central}\times i \times \text{Strike Step}_2$ for $i$ ranging from 1 to $\frac{N-3}{2}$.


#### <u>Offering</u>

Instrument:
* Volatility Index 10/25/50/75/100 (1s)
* Volatility Index 10/25/50/75/100 (2s)

Duration:
* 1 minute to 1440 minutes
* 1 hour to 24 hours
* 1 day to 365 days

### Product Pricing

Vanillas are priced using the Black Scholes Model:

$$
\begin{aligned}
Call &= e^{-r(T-t)}\left(S_t N(d_1)-KN(d_2) \right)\\
Put &= e^{-r(T-t)}\left(K N(-d_2)-S_tN(-d_1) \right)\\
\end{aligned}
$$

where $d_1 = \frac{\ln \left(\frac{S_t}{K}\right) + \frac{\sigma^2}{2}(T - t)}{\sigma \sqrt{T - t}}$ and $d_2 = d_1-{\sigma \sqrt{T - t}}$.

#### Bid-Ask Spread
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

## Validation Methodology
### Overview
<div align='justify'>
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

### Assumptions
All fiat currencies (USD, EUR, GBP, AUD) were tested however for cryptocurrencies, only BTC and ETH were tested as it was assumed that all cryptocurrencies will perform identically.

Most testing was done in the QA Box environment as many fixes were made ad hoc with BE team and deployed straight into our QA Box for testing. 

Testing for Back Office were done with the assumptions that reasonable values are keyed in. For example, "vol markup" should never be a negative value or that "Delta Config" should never be more than 1 or less than 0. 

### Validation Areas
1) Validate Barrier calculations
2) Validate Payout per Point
3) Validate Open Contracts' value (Current Bid Price)
4) Validate Expired Contracts' Payout
5) Validate Manually Sold Contracts Payout
6) Validate Implementation of Markups (Delta Charge and BS Markup)
7) Validate DTrader (Desktop and Mobile)
8) Validate Back Office functions

## Validation Results
### Results Statistic
Presentation slides also contain feedback from structuring quants on the observations found. 
| Scope | Presentation Slides | # Observations | Resolved | Non-blocker | Blocker |
| :- | :-: | :-: | :-: | :-:|:-:|
| <b>Total Observations </b>| - | 43 | 9 | 34 |-|
| Specification | [Link](https://docs.google.com/presentation/d/1H7nZ6hv7-pmix9w3nrCIbF6FRnqoipgoOZTlQdPh8lk/edit?usp=share_link)|2 | 2  | -|-|
| Backend Code | [Link](https://docs.google.com/presentation/d/1qKIQHHEewCd4snQMOTKeOLgIXDSgxyoXn8b8vbwlRiQ/edit?usp=share_link)|6 | 1|5 |-|
| DTrader |[Link](https://docs.google.com/presentation/d/1w_BdwqgWSWHRJj6iZBioXJboSsRXLUUTl-t4MNxm_uE/edit?usp=share_link)| 9 | 1|8|-|
| API | [Link](https://docs.google.com/presentation/d/1s7BIVQfAWTdYH4hqAUoVVN69PYBIguatWPazb990_SA/edit?usp=share_link)|10 | 4 | 6|-|
| DTrader (Mobile) | [Link](https://docs.google.com/presentation/d/18MNUGkMFk1eoAd9P27XM8J8ssTPxQb1lNEP1JpUe9oE/edit?usp=sharing)|14 | - | 14|-|
| Back Office | [Link](https://docs.google.com/presentation/d/1aGdofSNule1DBi5rfzbDyzTmzZOXM9DNIG4Fwp3tiXM/edit?usp=sharing)| 2 | 1 | 1 | - |

    
### Specifications
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 |The formula for PnL was incorrectly written in the [specifications doc](https://docs.google.com/document/d/1QveybJfkNJIusr_tS1JpmPY4NoaLSDRwH3c6QL1T7Vo/edit), it should be PnL = Closing Price - stake.| Resolved (17/04/2023)|
|2 |The formula for affiliate commission when a contract is manually closed was incorrect in the [affiliate commission document](https://docs.google.com/document/d/1Xl8yoDUCIogZptXu_TUpkDxbOVPn9BqTFEPvqe1LJ9c/edit). | Resolved (22/05/2023)|


### Backend Code    
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 |Currently the commission is charged using the delta multiplied by the spot spread and vega multiplied by the vol spread. This method can be inconsistent with other new products.  | Non-blocker |
|2 |Strikes for all vanilla contracts are calculated based on the rounded spot price instead of the actual spot price. The rounding of the spot price to calculate strikes is not mentioned in the specs document. |Non-blocker |
|3 |Strike steps for daily contracts are rounded to the nearest 10s in backend, this is not mentioned in the spec. |Non-blocker |
|4 |Central strikes are calculated just like every other strike and may not always be +0.00. However, Spec doc mentions that the ATM strike should have been set at +0.00 for the client to buy vanillas which have S = K.  |Non-blocker  |
|5 |Affiliate commission calculation when a contract is manually closed was: $Stake \times \frac{(mid - bid)}{mid}$ which did not match the [affiliate commission document](https://docs.google.com/document/d/1Xl8yoDUCIogZptXu_TUpkDxbOVPn9BqTFEPvqe1LJ9c/edit).| Resolved (23/05/2023) |
|6 |Rounding of Min Stake may result in being able to purchase contracts with a slightly lower number of contracts than the minimum set in BO. |Non-blocker  |

### DTrader
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 |Winning contract is displayed as lost. This bug is very rare and method to reproduce is unknown as of now.   | Non-blocker |
|2 |Client cannot buy a contract when the minimum stake is greater than the maximum stake. This issue can be easily reproduced in Vol 50 (1s) by selecting long duration and far in the money options.  | Non-blocker |
|3 |Contracts with small enough stakes will show the same payout per point for different strikes. This issue has been raised for vanilla options on financials.  |Resolved (23/05/2023) |
|4 |DTrader Contract details does not show the exit spot of a sold contract. This might be useful to a client looking to calculate his PnL. The exit spot is sent by backend but not shown by frontend. |Non-blocker |
|5 | It is not clear to clients that there is a commission for closing a contract before expiry. The payout per point is only applicable for contracts that reach maturity.|Non-blocker |
|6 |Buying a daily+ contract (2 days for example), the duration of the contract is not 48 hours but instead till the end of the 2nd day. e.g. Buying a 2-day contract on 8AM Monday, the contract does not expire on 8AM Wednesday but instead on the end of Wednesday (23:59:59 GMT +0).|Non-blocker |
|7 |When selecting the ATM strike, the visualisation displayed blocks the view of the current spot and the client needs to view the tiny display of the spot at the top left instead. |Non-blocker |
|8 |Inconsistent strike display style for Intraday and Daily+ contracts. |Non-Blocker |
|9 |Min stake error message may show min stake for call when put is selected and vice versa. |Non-Blocker |

### DTrader (Mobile)
It's not conducive to explain mobile issues with words alone, please refer to the slides [here](https://docs.google.com/presentation/d/18MNUGkMFk1eoAd9P27XM8J8ssTPxQb1lNEP1JpUe9oE/edit?usp=sharing) for a clearer understanding.
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 |Unable to enter numbers and no error pops up informing user when trying to key in numbers bigger than the max duration offered for the contract.  | Non-blocker |
|2 |Client can enter stake over 2,000 and error only shows after closing the keyboard. Which is inconsistent from max duration. | Non-blocker |
|3 |When tapping on duration to change the duration of contract the stake pops up as both duration and stake are the same button instead of being separated.   |Non-blocker |
|4 |Clients cannot instantly sell bought contracts on mobile as they have to wait for “Purchase confirmation” pop up to go away before being able to navigate to recent position tabs to sell their opened position.  |Non-blocker |
|5 | Selecting ATM strike blocks current spot display. |Non-blocker |
|6 |Pressing the "arrow down"	symbol hides the strike choice but not contract duration option which is inconsistent and has an unclear purpose. |Non-blocker |
|7 |Inconsistent strike price display between mobile and desktop. |Non-blocker |

Additional observations by structuring quants team on mobile. 
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 |Strike price are written as strike in some parts.   | Non-blocker |
|2 |Opening the Duration Tab for the first time will show incorrect expiry time.  | Non-blocker |
|3 |When tapping a point on the feed, the points are not selected correctly. |Non-blocker |
|4 |When zoomed in, the values on the x-axis changes every second.  |Non-blocker |
|5 | The placement of value of PnL in Trade Table and Statement can be enhanced.|Non-blocker |
|6 |Positions that are expired/sold cannot be cleared from the recent positions tab. |Non-blocker |
|7 |Strike price tooltip for duration more than 1 Day has different behaviour than intraday. |Non-blocker |

### Back Office
|No. | Area | Observations(s) | Status | 
| :- | :- | :- | :- | 
|1 | Min Number Of Contracts	| It is possible to buy a contract with number of contracts slightly below this value. Refer to the Backend Code section for further details.  | Non-Blocker |
|2 | Global Potential Loss	| Incorrectly implemented as all vanilla contracts have unrealised PnL of 0 in backend | Limit has been removed from vanilla contracts | Resolved |

### API
#### Final round of validation on 12 July 2023
<div align='justify'>
 
To review the following card: https://redmine.deriv.cloud/issues/95135#Enable_vanilla_synthetics_on_real. During review, a bug where selling using api would result in selling a contract at 0 regardless of actual contract value. The following card which fixes the bug was validated as well: https://redmine.deriv.cloud/issues/96450#%22sell%22_API_call_sometimes_doesn't_sell_at_market. Below Statistics are from testing in the QA box environment where the fix was already deployed. 

</div>

|No.| Area | Test Cases | Pass | Fail |
| :- | :- | :-: | :-: |  :-: | 
|1| Barrier Calculations | 145 | 145 | 0 | 
|2| Payout per Point | 100 | 100 | 0 | 
|3| Contract Value in Price Proposal | 110 | 110 | 0 | 
|4| Contract Value for Manually Sold Contracts | 135 | 135 | 0 |
|5| Contract Value for Expired Contracts | 100 | 100 | 0 | 

|No. | Area| Observations(s) | Status |
| :- | :- |:- | :- |
|1| Contract Value for Manually Sold Contracts | Selling contracts through the API will occasionally sell the contract at 0 regardless of actual contract value. | Resolved (12/07/2023) |

#### Re-validation of API on 22 June 2023
<div align='justify'>
 
This round of API testing was done to validate [a fix](https://redmine.deriv.cloud/issues/95653#Inconsistent_sell_at_market_tick_for_vanilla_and_turbos) for the issue of returning the wrong exit spot in the API. Issue considered resolved as there were no occurrences of the issue after a fix was deployed and tested in QA141. The 4 fail cases are due to another pre-existing observation where the contract duration returned in the API may differ from the actual contract duration by 1 or 2 seconds depending on the actual sell time.
 
</div>

|No.| Area | Test Cases | Pass | Fail |
| :- | :- | :-: | :-: |  :-: | 
|1| Barrier Calculations | 100 | 100 | 0 | 
|2| Payout per Point | 100 | 100 | 0 | 
|3| Contract Value in Price Proposal | 100 | 98 | 2 | 
|4| Contract Value for Manually Sold Contracts | 100 | 98 | 2 |
|5| Contract Value for Expired Contracts | 100 | 100 | 0 | 


#### Previous Observations (23 May 2023)
|No.| Area | Test Cases | Pass | Fail |
| :- | :- | :-: | :-: |  :-: | 
|1| Barrier Calculations | 50 | 50 | 0 | 
|2| Payout per Point | 270 | 267 | 3 | 
|3| Contract Value in Price Proposal | 120 | 87 | 33 | 
|4| Contract Value for Manually Sold Contracts | 113 | 87 | 26 |
|5| Contract Value for Expired Contracts | 50 | 50 | 0 | 

|No. | Area| Observations(s) | Status |
| :- | :- |:- | :- |
|1| Payout per Point |  The API may display an entry spot that differs from the actual pricing spot due to system processing time, a fix has been deployed to reduce this issue significantly. |Non-blocker |
|2| Contract Value in Price Proposal |  The contract duration in the API may differ from the actual contract duration by 1 or 2 seconds depending on the actual sell time, but the payout calculation is still correct.  |Non-blocker|
|3| Contract Value for Manually Sold Contracts |  When client sells back the contract, the pricing spot in the bid price calculation may differ from the exit spot shown in the API (DTrader does not show the exit spot at this moment).  |Resolved (22/06/2023)|
|4| Contract Value for Manually Sold Contracts |  The contract duration in the API may differ from the actual contract duration by 1 or 2 seconds depending on the actual sell time, but the payout calculation is still correct.  |Non-blocker |


#### Other Observations
The fail cases from these other issues are not included in the current observation statistics.
|No. | Area| Observations(s) | Status |
| :- | :- |:- | :- |
|1| Barriers | For contracts with a duration of less than 5 minutes on 2s Indexes, barriers are calculated using the value of dt minus 2 seconds. [Justification from BE](https://github.com/regentmarkets/perl-Finance-Contract/blob/master/lib/Finance/Contract.pm#L822)| Non-blocker |
|2| Payout per Point | The API may display an entry spot that differs from the actual pricing spot. For intraday contracts, barrier returned by API will be used to calculate the payout  but the initial PPP was calculated from a different barrier. | Resolved (23/05/2023) |
|3| Payout per Point | For contracts less than 5 minutes on 2s Indexes, the PPP is calculated using dt minus 2 seconds. | Non-blocker |
|4| Contract Value for Manually Sold Contracts | Selling Crypto contracts through the API, they will be sold at a price of 0 regardless of the actual contract value when attempting to sell at market price. | Resolved (08/05/2023) |
|5| Markups | Delta Charge markup was incorrectly implemented. | Resolved (23/05/2023) |

### Appendix
| Document | Link | 
| :- | :- | 
| Specifications | https://docs.google.com/document/d/1QveybJfkNJIusr_tS1JpmPY4NoaLSDRwH3c6QL1T7Vo/edit?usp=sharing |
| Affiliate Commission | https://docs.google.com/document/d/1Xl8yoDUCIogZptXu_TUpkDxbOVPn9BqTFEPvqe1LJ9c/edit?usp=sharing |

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.