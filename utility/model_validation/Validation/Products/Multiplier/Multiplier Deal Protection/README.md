# Deal Protection feature for Multipliers Validation Report 
#### Updated 16/01/2024

### <u>Person In Charge</u>
|Area | Person In Charge | 
| :- | :- | 
| Product Owner | Alassane (alassane@deriv.com) | 
| Backend | - | 
| QA | - | 
| Model Validation | Matthew (matthew.chan@deriv.com) / Kuan Lim (kuanlim@deriv.com) |

## Summary
<div align='justify'>

The validation process for the new deal protection feature for Multipliers covers validating the product specifications, PQ cross check and backend product specifications. This validation also involves comparing the closed-form formula for deal protection with a Monte Carlo pricer. There are a total of 11 observations found, of which there is 1 blocker, 6 non-blockers and 4 resolved observations.

The current blocker is that there is no commission on the premium for deal protection, exposing the company to exploits and leaving the theoretical revenue from deal protection to be 0. 

Report in the form of slides [here](https://docs.google.com/presentation/d/1tyJjT5VM18RcUctTLvLcoSA7o9UcWP16FAr6W4oJFgY/edit?usp=sharing)

</div>

## Product Information

#### <u>Product Description</u>
<div align='justify'>
The buyer/seller of the multiplier option will participate in any increase/decrease above/below the initial spot price. However, the client assumes the risk that their multiplier option may be terminated when they lose their entire stake. <br><br>

The Deal Protection feature allows the client to withstand the loss of their entire stake for a selected duration. With the deal protection feature, the multiplier option remains active even if the stop-out level is breached.

Clients can buy Deal Protection as many times as they want. The only condition is that the  current spot price has to be above the Stop level for Multiplier Up (MULTUP) and below the stop out level for Multiplier Down (MULTDOWN). 

</div>

#### <u>Term & Conditions</u>

 * Before a stop-out level is reached, client selects the Deal Protection feature and and chooses a duration. 
 
 * A premium corresponding to the price of the deal protection is deducted from the client's balance.
 
 * The client can close his position anytime.

#### <u>Offering</u>

Instrument:
* Volatility Index 10/25/50/75/100/150/250 (1s)
* Volatility Index 10/25/50/75/100 (2s)

Duration:
* 1 minute to 60 minutes 
* 1 hour to 24 hours 
* 1 day to 365 days

### Product Pricing

Premium for deal protection is priced using down-and-in put (DIP) for MULTUP and up-and-in call (UIC) for MULTDOWN.

#### For MULTUP
$$
\begin{aligned}
DIP &= -SN(-x_1)e^{-qT}+Ke^{-rT}N\left(-x_1+\sigma\sqrt{T}\right)+Se^{-qT}\left(\frac{H}{S}\right)^{2\lambda} \left[ N(y)-N(y_1) \right] -Ke^{-rT}\left(\frac{H}{S}\right)^{2\lambda-2}\left[ N\left(y-\sigma\sqrt{T}\right)-N\left(y_1-\sigma\sqrt{T}\right) \right] 
\end{aligned}
$$

#### For MULTDOWN
$$
\begin{aligned}
UIC &= SN(-x_1)e^{-qT}-Ke^{-rT}N\left(x_1-\sigma\sqrt{T}\right)-Se^{-qT}\left(\frac{H}{S}\right)^{2\lambda} \left[ N(-y)-N(-y_1) \right] +Ke^{-rT}\left(\frac{H}{S}\right)^{2\lambda-2}\left[ N\left(-y+\sigma\sqrt{T}\right)-N\left(-y_1+\sigma\sqrt{T}\right) \right] 
\end{aligned}
$$

where,
* $r = \text{risk free rate}$
* $q = \text{dividend yield}$
* $\lambda = \frac{r-q+\sigma^2/2}{\sigma^2}$
* $y = \frac{\ln{\frac{H^2}{SK}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t}$
* $x_1 = \frac{\ln{\frac{S}{H}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t}$
* $y_1 = \frac{\ln{\frac{H}{S}}}{\sigma\sqrt{t}}+\lambda\sigma\sqrt{t} $

Finally,
* $\text{Deal Protection Premium} \_{MULTUP|MULTDOWN} =  \text{Premium} {}_{DIP|UIC} \times \frac{\text{Stake} \times \text{Multiplier}}{\text{Initial Spot}}$

## Validation Methodology
### Overview
<div align='justify'>
Pricing of premium for deal protection was validated on all indices using their approximate spot price at the time of validation, min/max multiplier and min/max stake. Duration for each contract was chosen based on min and max duration offered for deal protection. Further testing on edge cases was also done to evaluate what minimum stake/duration of deal protection was required to meet the minimum premium of $1. 

To ensure the pricing formula is accurate, its pricing is also compared to a Monte Carlo pricer.
</div>

### Validation Areas
1) Validate pricing methodology 
2) Comparison of the closed form pricer against Monte Carlo pricer
3) Test edge cases based on current spot prices 

## Validation Results
### Results Statistic
Observations also available in [slides form](https://docs.google.com/presentation/d/1tyJjT5VM18RcUctTLvLcoSA7o9UcWP16FAr6W4oJFgY/edit?usp=sharing).
| Scope | # Observations | Resolved | Non-blocker | Blocker |
| :- | :-: | :-: | :-:|:-:|
| <b>Total Observations </b> | 11 | 5 | 5 | 1 |
| Product Specification | 3 | 2  | 1 | - |
| Backend Product Specifications | 7 | 2 | 4 | 1 |
| PQ Cross Check | 1 | 1 | - | - |

### Product Specifications
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Symbol B is used twice to represent different variable in pricing formula (Stop-out level (the barrier) as well as the Vanilla Option formula). | Resolved (06/12/2023)|
|2 | Symbol E is used in pricing but not defined. | Resolved (06/12/2023)|
|3 | Dividend rate should be negative instead of positive in all formula. | Non-blocker |

### Backend Product Specifications
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Specifications do not provide a commission structure for pricing deal protection. | Blocker |
|1 | Specifications for Deal Protection extension is not clear. | Resolved (08/01/2024)|
|2 | Deal Protection Premium formula confusing. Unclear Which “S”, St or S0 to use in premium (A, N, C, D) calculations formula. | Resolved (28/12/2023)|
|3 | Dividend rate should be negative instead of positive in all formula. | Non-blocker |
|4 | Incorrect Commission used in Multiplier Formula. Commission in formula should be in percent form, not dollar form. i.e. Commission for multiplier: 0.002% | Non-blocker |
|5 | Incorrect numbers in scenario analysis section. e.g. correct payout for multiplier should be 61.18, but 61.68 is listed. | Non-blocker |
|6 | Incorrect calculations in unit test section. | Non-blocker |

### PQ Cross Check
|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Incorrect formula used for number of contracts in PQ cross check. Number of contracts should be multiplier x stake / entry price instead of multiplier x stake / stop out price | Resolved (28/12/2023)|

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.
