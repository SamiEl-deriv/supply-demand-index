# BO on DEX Validation

Last updated: 20 November 2024

### Squad Members

| Area             | Person In Charge                                    |
| :--------------- | :-------------------------------------------------- |
| Product Owner    | Antoine Mille (antoine.mille@regentmarkets.com)   |
| Project Manager  | Ekaterina (ekaterina@regentmarkets.com)|
| Backend          | -   |
| QA               |     -       |
| Model Validation | Matthew (matthew.chan@regentmarkets.com)                    |

## Summary

The validation process for BO on DEX indices covers Backend Product Specification, including validating the pricing model.

<!-- 
Contract parameters such as turboslong/turbosshort, duration, stake, and take profit were chosen at random while all the volatility indices are tested. Testing on risk monitoring tools in Back-Office were also conducted. Spread parameters, which set in Backoffice, are also validated. -->


Total observations:

- 4 observations found.
  <!-- - 4 Resolved. -->

Report in the form of slides [here](https://docs.google.com/presentation/d/1yKbTEA7EMs7ueAsUI2YsS8pWZj-PSgD67mx607xeQBo/edit?usp=sharing). 
Working file for the validation process can be found in the same GitHub Directory as this report.

## Product information

To expand the scope of Binary Options (BO) trading by introducing Rise/Fall and Higher/Lower to Double Exponential Jump (DEX) Indices and Hybrid Indices. These indices exhibit sudden, significant price jumps, allowing traders to capitalize on rapid price movements, unlike traditional, smooth price changes. The pricing of Rise/Fall contracts in these Binary Options uses the strike price set equal to the spot price.


#### Underlying Symbols

- DEX 600 DN
- DEX 900 DN
- DEX 1500 DN
- DEX 600 UP
- DEX 900 UP
- DEX 1500 UP
<!-- Hybrid Indices ? -->



#### Offerings
- 1 tick to 1 year
- min stake = $0.35
- max payout = $50,000


### Product Pricing Replication 
<!-- We can use Black-Scholes Model for Down-and-Out Call option and Up-and-Out Put option to price Turbos (refer appendix for derivation). The <b>close form formula for turbos without comission</b> is given by:

Long Turbos (Down-and-Out Call):
  
$$
\text{S}-\text{K}
$$

Short Turbos (Up-and-Out Put):
  
$$
\text{K}-\text{S}
$$

#### Commission Model

Add 1.2% markup to contract unit price F. 
$$
\begin{align*}
F &= \text{unit mid price for 1 contract}\\
\text{Payout} &= \frac{\text{Stake}}{F+ 1.2\%} 
\end{align*}
$$ -->


#### Terms & Conditions

Client pays:

$$
\text{Stake}
$$

Client gets:
  - for Rise:

$$
\begin{align*}
\text{Payout if exit spot is above entry spot else 0}
\end{align*}
$$
  - for Fall:

$$
\begin{align*}
\text{Payout if exit spot is below entry spot else 0}
\end{align*}
$$

## Validation Methodology
### Overview

Pricing model was validated using Monte carlo in addtion to evaluating the formulas.

  
#### Validation Areas
- Backend Product Specification
  - Pricing 

### Assumption

<!-- Most testing is done in the QA Box environment as many fixes were made ad hoc with BE team and deployed straight into our QA Box for testing. Hence, the assumption is the production should be (almost) the same as the test environment. -->


## Validation Findings

### Summary of Findings

| Scope                 | Slide | # Observations | Resolved | Non-blocker<br>(Future improvements) | Blocker |
| :-------------------- | :----:| :------------: | :------: | :----------------------------------: | :-----: |
| <b>Total Observations | - |      <b>4     |    4     |                  -                  |    -    |
| BE Specification         |[link](https://docs.google.com/document/d/1ak-Fk8FOlnrht2-EnPLhMP8O4K2u0bqtvSWXAHL9GOs/edit?usp=sharing) |      4        |    4     |                  -                   |    -    |
<!-- | DTrader               |[link](https://docs.google.com/presentation/d/1WuaGRFPpPNwraa_7WE2lIwBL8YU5M76pDlUUznuNnWU/edit?usp=sharing) |       18       |    9     |                  9                  |    -    |
| DTrader Mobile        |[link](https://docs.google.com/presentation/d/1ASUuyP7T_Uc7qwlmfZ4B8NhrjHeKOkBPaI73Naq1LU8/edit?usp=sharing) |      5        |    4     |                  1                   |    -    |
| API                   |[link](https://docs.google.com/presentation/d/1g6EwbmNMssIYkXIX4GYfIP0Sy0jMmpx6Fe2ns9CSor8/edit?usp=sharing) |       1        |    -     |                  1                   |    -    | -->

## Specification

Specification issues are listed [here](https://docs.google.com/presentation/d/1yKbTEA7EMs7ueAsUI2YsS8pWZj-PSgD67mx607xeQBo/edit?usp=drive_link).

No. | Issue(s) | Status |
| :- | :- | :- |
|1 | DEX 1200 should be replaced with DEX 1500. | Resolved |
|2 | Inaccurate formula for characteristic equation| Resolved |
|3 | Inaccurate formula for homogeneous equation| Resolved |
|4 | Inaccurate formula for laplace price of BO| Resolved |


#### Reference
Sepp, Artur. (2003). Analytical Pricing of Double-Barrier Options under a Double-Exponential Jump Diffusion Process: Applications of Laplace Transform.

R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.
