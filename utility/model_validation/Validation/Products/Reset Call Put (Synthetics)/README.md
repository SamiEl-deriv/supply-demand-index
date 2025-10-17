# Reset Call/Put Validation Report 
#### Updated 30/10/2023 

### <u>Person In Charge</u>
There is no squad for this product
|Area | Person In Charge | 
| :- | :- | 
| Product Owner | Alassane (alassane@deriv.com) / Tahar (tahar@deriv.com) | 
| Backend | Jun Bon (junbon@deriv.com) / Tanvir Faisal (tanvir@deriv.com) | 
| QA | Yashwini (Yashwini@deriv.com) | 
| Model Validation | Matthew (matthew.chan@deriv.com) |

## Summary
<div align='justify'>
During prior documentation efforts for existing products, Kuan Lim identified an error in the existing pricing formula of Reset Call/Put. This validation involves comparing the enhanced closed-form formula with a Monte Carlo pricer and also validating the payout of contracts obtained through the API with MV calculated value. There is only 1 observation found that the pricing of contracts with duration of odd number of ticks are incorrect. This observation is a blocker which needs to be resolved before release.
<br><br>
Update 30/10/2023: Blocker has been resolved. Incorrect pricing for contracts with odd number of ticks has been resolved by using the correct reset time in formula.<br>
 
<br>Report in the form of slides [here](https://docs.google.com/presentation/d/1oqntcdwL-5BAeS7cxK8dDTLYZwEaqVH4KwMh9dHg7UU/edit#slide=id.g242917dc3c0_0_6)
 
</div>

## Product Information

#### <u>Product Description</u>
Perl code version: https://redmine.deriv.cloud/issues/89732#enhancement_to_reset_pricing_engine. <br>
<div align='justify'>
When client buys a Reset Call or Reset Put, the client predicts whether the exit spot will be higher or lower than either the entry spot or the spot at reset time. Note that if the exit spot is equal to the entry spot or the spot at reset time, the client will not receive the payout.<br>
Client can choose the duration of the contract and the stake amount. When the client's prediction is correct, the client will receive a payout, otherwise the client will lose their stake. Reset time is automatically set by the system at around half the expiry time. E.g. 8 seconds contract has reset time at 4 seconds and 5 seconds contract have reset time at 2 seconds. 
 
</div>

#### <u>Term & Conditions</u>

Client Pays: 
$$Stake$$

Client Receives:
|Contract Type| Term & Conditions |
| :- | :- | 
|Reset Call|  At the end of contract, the client will receive the payout, if the exit spot is strictly higher than either the entry spot or the spot at reset time or lose the stake, if otherwise.| 
|Reset Put| At the end of contract, the client will receive the payout, if the exit spot is strictly lower than either the entry spot or the spot at reset time or lose the stake, if otherwise. | 

where

* $\text{Payout} = \frac{\text{Stake}}{\text{Reset}_{Ask}}$
* $\text{Reset}_{Ask} = \text{Base Prob + Commission}$

#### <u>Offering</u>

Instrument:
* Volatility Index 10/25/50/75/100 (1s)
* Volatility Index 10/25/50/75/100 (2s)

Duration:
* 5 to 10 ticks
* 1 to 120 minutes
* 1 to 2 hours

### Product Pricing

Reset Call/Put are priced using a closed form formula:

$$
\begin{aligned}
Reset\space Call\space Base\space Prob &= P(S_T\ge S_t, S_t\le K)+P(S_T\ge K,S_t\ge K)\\
&=P(S_T\ge S_t)\times P(S_t\le K) + P(S_T, S_t > K) \\
&=N(-d_{2,t})\times N(d_{2,T-t}) + bivariate\left(d_{2,T},d_{2,t},\rho=\sqrt{\frac{t}{T}}\right)\\\\
Reset\space Put\space Base\space Prob &= P(S_T\le S_t, S_t\ge K)+P(S_T\le K,S_t\le K)\\
&=P(S_T\le S_t)\times P(S_t\ge K) + P(S_T, S_t < K) \\
&=N(d_{2,t})\times N(-d_{2,T-t}) + bivariate\left(-d_{2,T},-d_{2,t},\rho=\sqrt{\frac{t}{T}}\right)\\\\
Reset_{Ask} &= e^{-r\\_ q\cdot t}\cdot Reset\space Base\space Prob + Commission\\
\end{aligned}
$$

Where, 
* $d_{2,\tau} = \frac{-\frac{\sigma^2}{2}\tau}{\sigma\sqrt{\tau}}$.
* $bivariate(x, y, \rho) \text{ evaluates the lower left tail of the bivariate normal distribution}$
* $r\\_ q =0\text{ For intraday contracts}$

Finally,
* $\text{Payout} = \frac{\text{Stake}}{\text{Reset}_{Ask}}$

#### Commision Structure
An additional charge of 1.2 % (At time of validation) on top of the base probability of the contract is added for all Reset Call/Put contracts.

## Validation Methodology
### Overview
<div align='justify'>
Pricing of Reset Call/Put were validated on all indices. Duration for each contract was randomly chosen while ensuring that min and max durations were also selected to validate edge cases. Commission values were assumed to be their default values (1.2%). The payout of the contracts were validated by comparing the price proposal payout value to MV calculated value. <br>
<br>
To ensure the enhanced closed-form pricing formula is accurate, its pricing is also compared to a Monte Carlo pricer. 5000 randomly generated contracts were priced with both the closed form formula and also the Monte Carlo pricer resulting in a mean difference of 0.04 %. The backend's bivariate function was also validated by comparing its value to the MV calculated value. 
 </div>

### Validation Areas
1) Validation of [Backend Bivariate CDF function](https://github.com/regentmarkets/perl-Pricing-Engine-Reset/blob/2c70b4f2a6861923e5a64a9eee3038af84015a94/lib/Pricing/Engine/Reset.pm#L69)
2) Comparison of the closed form pricer against Monte Carlo pricer
3) Validation of payout calculations from API


## Validation Results
### Results Statistic
There is only 1 observation for this product. Results from additional testing are also included in this section. 
| Scope | Presentation Slides | # Observations | Resolved | Non-blocker | Blocker |
| :- | :-: | :-: | :-: | :-:|:-:|
| Pricing from API | [Link](https://docs.google.com/presentation/d/1oqntcdwL-5BAeS7cxK8dDTLYZwEaqVH4KwMh9dHg7UU/edit#slide=id.g242917dc3c0_0_6)| 1 | 1 | -|-|

### Pricing from API
Payout from API was compared to MV calculated payout. 27 fail cases were from contracts having an odd number of ticks as the payout is calculated using incorrect reset time. E.g. For contract duration of 5 ticks, the reset time (automatically set by the system) is 2 ticks but the contract is priced with 2.5 ticks reset time. 

|No.| Area | Test Cases | Pass | Fail |
| :- | :- | :-: | :-: |  :-: | 
|1| Payout | 200 | 173 | 27 | 

|No. | Observations(s) | Status |
| :- | :- | :- |
|1 | Incorrect pricing for contracts with odd number of ticks. | Resolved (30/10/2023)|

#### Post Reset Time Fix Validation 
Validation of Fix for Reset Time was done on 30/10/2023 <br>
card: https://app.clickup.com/t/20696747/OPT-460

|No.| Area | Test Cases | Pass | Fail |
| :- | :- | :-: | :-: |  :-: | 
|1| Payout | 150 | 150 | 0 | 

### Additional Tests  
* Backend bivariate function produces identical results to MV version. 
* Closed form formula produces almost identical prices to Monte Carlo pricer even at much longer durations (1 year) even though we only offer up to 2 hour contracts. Highest difference between Monte Carlo Pricer and closed form pricing is 0.2% while the mean difference from 5000 randomly generated contracts is 0.04%.

|No.| Area | Test Cases | Statistic |
| :- | :- | :-: | :-: | 
|1| [Backend Bivariate CDF](https://github.com/regentmarkets/perl-Pricing-Engine-Reset/blob/2c70b4f2a6861923e5a64a9eee3038af84015a94/lib/Pricing/Engine/Reset.pm#L69) against MV version | 50 | 50 Passed | 
|2| Closed form pricer against Monte Carlo pricer | 5000 |  Mean Diff: 0.04% | 

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.