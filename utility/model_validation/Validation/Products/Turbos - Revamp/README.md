# Turbo - Revamp Validation Report

Latest update - 31st July 2024

### <u>Squad Members</u>

| Area             | Person In Charge                                    |
| :--------------- | :-------------------------------------------------- |
| Squad Leader     | Yng Shan (yngshan@deriv.com)                        | 
| Scrum Master     | Ekaterina (Kate) (ekaterina@deriv.com)              | 
| Backend          | Qasim (qasim@deriv.com)                             | 
| QA               | Luqman (luqman@deriv.com)                           | 
| Model Validation | Matthew (matthew.chan@deriv.com)                    | 

## Summary

The validation process for Turbos Revamp on Volatility indices covers API testing, and Backend Specification, including but limited to validating the pricing model with spread, the implementation of calculating the allowable PPP choices, and the overall implementation of the revamped version.

Contract parameters such as TURBOSLONG/TURBOSSHORT, duration, stake, and PPP choice were chosen at random while all the volatility indices are tested. 


Total observations:

- 6 issues found.
  - 4 Resolved.
  - 2 Non-Blocker.

Report in the form of slides [here](https://drive.google.com/drive/folders/1sNg_IylrD0oCY0J-04i0u6-LExICft3f). 
Working file for the validation process [here](https://github.com/regentmarkets/quants-model-validation/tree/master/Other%20Projects/API%20Testing%20Automation%20Project).

## Product information

The buyer of the Long/Short Turbo will participate in any increase/decrease above/below a predetermined barrier.

However the client bears the risk that the turbo option will be knocked out and early terminated if the spot price touches or breaches a predetermined barrier.

In a Turbo Option, the <b>knock-out barrier, $B$ and the strike, $K$ are equal</b> and the options are offered in the money at the start of the trade. 


### Underlying Offerings

- Volatility 10, 25, 50, 75, 100 (1s) Index
- Volatility 10, 25, 50, 75, 100 Index


### Duration

- 5 to 10 ticks
- 15 seconds to 60 seconds
- 1 minute to 60 minutes
- 1 hour to 24 hours
- 1 day to 365 days


### Product Replication and Close Form Formula
#### Black-Scholes Pricing

We can use Black-Scholes Model for Down-and-Out Call option and Up-and-Out Put option to price Turbos (refer appendix for derivation). The <b>close form formula for turbos without comission</b> is given by:

Long Turbos (Down-and-Out Call):
  
$$
\text{S}-\text{K}
$$

Short Turbos (Up-and-Out Put):
  
$$
\text{K}-\text{S}
$$

#### Spread Model


|             | Long Turbos | Short Turbos |
| :---------: | :---------: |:----------: |
| Enter Trade |  Ask - $B$  | $B$ - Bid |
| Close Trade |  Bid - $B$  | $B$ - Ask |



where,

$$
\text{Ask} = S_{t} \times (1 + \text{Ask Spread})
$$

$$
\text{Bid} = S_{t} \times (1 - \text{Bid Spread})
$$

and,

$$
\begin{aligned}
\text{Ask Spread} &= \text{Average Tick Size Up in Fixed Table} \times \text{Ticks Commission Up in BO} \\
\text{Bid Spread} &= \text{Average Tick Size Down in Fixed Table} \times \text{Ticks Commission Down in BO} 
\end{aligned}
$$

Remark: We only charge a commission when entering or closing the contract. Thus, there is **no commission when the contracts expire at expiry date**.

#### Terms & Conditions

Client pays:

$$
\text{Stake}
$$

Client's Payout:
  - for Long Turbos:

$$
\begin{align*}
\text{Payout} &=
\begin{cases}
n \times (\text{Bid}-B) &\text{Manually Sell/Take Profit}\\
n \times (S_t-B) &\text{Expired Contract}\\
0 &\text{Event(Knock Out)}
\end{cases}\\\\
\end{align*}
$$
- for Short Turbos:

$$
\begin{align*}
\text{Payout} &=
\begin{cases}
n \times (B-\text{Ask}) &\text{Manually Sell/Take Profit}\\
n \times (B-S_t) &\text{Expired Contract}\\
0 &\text{Event(Knock Out)}
\end{cases}\\\\
\end{align*}
$$


### Revamp Changes
New logic to allow cleints to choose a PPP from a list of available PPP choices instead of choosing a barrier. Barrier will then be calculated based on the chosen PPP. More details on the changes can be found [here](https://github.com/regentmarkets/quants/blob/master/backend_product_specification/Turbos/Turbos%20BE%20Specs.md).


## Validation Methodology
### Overview

All indices that offers Turbos were validated along with all time offerings. The indices, the barrier choice, and the contract duration were randomly selected when purchasing contract using API. The commission parameter were set according to the Backend specification.

  
#### Validation Areas
- Backend Specification
  - Contract Specific Configuration Tool
  - Unit Tests 
- API Validation
    - Implementation of revamp on API
    - Payout per Point Choices
    - Pricing of contracts
    - Changes to Proposal
    - Sell price (Manually & at Expiry)
- Back Office Tools Validation
    - Turbo PPP choices configuration
    - Maximum number of open positions per instrument per client

### Assumption

Most testing is done in the QA Box environment as many fixes were made ad hoc with BE team and deployed straight into our QA Box for testing. Hence, the assumption is the production should be (almost) the same as the test environment.


## Validation Findings

### Summary of Findings

| Scope                 | Slide | # Observations | Resolved | Non-blocker<br>(Future improvements) | Blocker |
| :-------------------- | :----:| :------------: | :------: | :----------------------------------: | :-----: |
| <b>Total Observations | - |      <b>6     |    4     |                  2                  |    -    |
| Backend Specification         |[link](https://docs.google.com/presentation/d/1WAaFFcXP3EwOLco7KFafsKzwCWQYw5Og9vKqFKL98Cs/edit?usp=sharing) |      1        |    1     |    -     |    -    |
| API                   |[link](https://docs.google.com/presentation/d/1Kgd1rYvOFIhgT0YnX2dTIaYo6Otj1lrYzi0N7qoCcfw/edit?usp=sharing) |       5        |    3     |       2     |    -    |

### Backend Specification

Specification Observations are listed [here](https://docs.google.com/presentation/d/1WAaFFcXP3EwOLco7KFafsKzwCWQYw5Og9vKqFKL98Cs/edit?usp=sharing).

No. | Issue(s) | Status |
| :- | :- | :- |
|1 | Incorrect number of contracts in unit test | Resolved |


### API
API Observations are listed [here](https://docs.google.com/presentation/d/1Kgd1rYvOFIhgT0YnX2dTIaYo6Otj1lrYzi0N7qoCcfw/edit?usp=sharing).

|No. | Issue(s) | Status |
| :- | :- | :- |
|1 |Ability to buy invalid TURBOSSHORT contract (barrier < entry spot).| Resolved |
|2 |Unable to purchase contract with no clear error code.| Resolved |
|3 |Unable to use “buy contract request” to purchase turbos if parameters are passed.| Resolved |
|4 |‘payout_choices’ may not be suitable term in return proposal.| Non-Blocker |
|5 |Condition for ‘is_expired’ to be marked as ‘1’ is not clear in POC.| Non-Blocker |

## Appendix
| Document | Link | 
| :- | :- | 
| BE Specifications | https://github.com/regentmarkets/quants/blob/master/backend_product_specification/Turbos/Turbos%20BE%20Specs.md |

<br><br>
R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities.

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.
