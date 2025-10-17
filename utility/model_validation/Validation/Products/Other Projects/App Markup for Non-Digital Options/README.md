# App Markup for Non-Digital Option Validation Report
#### Updated 31/07/2024

### <u>Squad Members</u>
|Area | Person In Charge | 
| :- | :- | 
| Squad Leader | Yng Shan (yngshan@deriv.com) | 
| Scrum Master | Ekaterina (Kate) (ekaterina@deriv.com) | 
| Backend | Kok Jun Bon (junbon@deriv.com) | 
| QA | Luqman (luqman@deriv.com) | 
| Model Validation | Matthew (matthew.chan@deriv.com) \ Aaryan (aaryan@deriv.com) |

## Summary
<div align='justify'>
The validation process for the implementation of third-party app markups for non-Digital contracts was conducted for 4 products: accumulators, multipliers, vanilla options, and turbos. This validation ensures that the backend specification and implementation by the backend team are accurate.
   
<br>During the validation process, all 4 products and their respective features and parameters were thoroughly reviewed. Random selections of contract parameters, underlying assets, and duration were used to ensure comprehensive coverage.

Total observations:
- 12 issues found.
   - 12 Resolved.
   - 0 Blockers.
   - 0 Non-Blocker.
     
Report in the form of slides [here](https://docs.google.com/presentation/d/1eyysrwPxwLRrDzfWZKvF7yMOr2w5NE-H-IBhjfjmH_A/edit?usp=sharing)

</div>

## Product Information
### <u>Product Description</u>
The objective of this project was to introduce app-markup to non-Digital options (Accumulator, Multiplier, Vanilla, Turbo). Third-party app markup is a charge imposed and earned by third-party developers. 

### <u>Offering</u>
- Accumulator
- Multiplier
- Vanilla
- Turbo

## Validation Methodology
### Overview
The validation process for this task involved the preparation of unit tests by our team for use by the Backend team, validation of Backend Specifications, and the overall implementation of third-party app markups which included four products (Accumulator, Multiplier, Vanilla, and Turbo). Each product, along with its respective features and parameters, was thoroughly validated, including but limited to:

### Validation Areas
#### Overall
  1) Ensuring the implementation of app markup did not disrupt any existing API functionality.
  2) Verification that the app markup is correctly returned in the "Proposal" API response.
  3) Verification that the app markup is accurately reflected in the "Proposal Open Contract" API response.

#### Product Specific Validation
Validation of the calculation and application of app markup in each contract type.
- Accumulator
  1) App-markup being reflected in final stake amount
  2) Take profit calculations
  3) Sell price calculations
  4) PnL calculations
  5) Growth amount after each tick
- Multiplier
  1) App-markup being reflected in final stake amount
  2) Longcode accurately displays multiplier effect 
  3) Take profit barrier calculations
  4) Deal Cancellation pricing 
  5) Deal cancellation refund amount takes into account app-markup
  6) Stop out barrier calculations
  7) Stop loss barrier calculations
- Turbo
  1) App-markup being reflected in final stake amount
  2) Offered barriers calculations
  3) PPP calculations
  4) Bid price calculations
  5) Sell price calculations
  6) Expiry payout calculations
  7) PnL Calculations
- Turbo: Revamp
  1) App-markup being reflected in final stake amount
  2) PPP choices calculations
  3) Barrier calculations
  4) Bid price calculations
  5) Sell price calculations
  6) Expiry payout calculations
  7) PnL Calculations 
- Vanilla
  1) App-markup being reflected in final stake amount
  2) Offered barriers calculations
  3) PPP calculations
  4) Bid price calculations
  5) Sell price calculations
  6) Expiry payout calculations
  7) PnL Calculations 


### Assumptions
It was assumed that the implementation of third-party app markup into non-Digital contracts would not break previously validated product-specific backoffice configurations and parameters. Consequently, individual backoffice tools were not tested separately but were instead evaluated collectively during the overall product testing process.

Most testing is done in the QA Box environment as many fixes were made ad hoc with the BE team and deployed straight into our QA Box for testing. Hence, the assumption is the production should be (almost) the same as the test environment.


## Validation Results
### Results Statistic
| Scope | Presentation Slides | # Observations | Resolved | Non-blocker | Blocker |
| :- | :-: | :-: | :-: | :-:|:-:|
| <b>Total Observations </b>| - | 12 | 12 | - |-|
| BE Specification | [Link](https://docs.google.com/presentation/d/1eyysrwPxwLRrDzfWZKvF7yMOr2w5NE-H-IBhjfjmH_A/edit#slide=id.g2ed7c95bc58_0_36)|1 | 1  | -|-|
| API | [Link](https://docs.google.com/presentation/d/1eyysrwPxwLRrDzfWZKvF7yMOr2w5NE-H-IBhjfjmH_A/edit?usp=sharing)|11 | 11 | -|-|


### BE Specification
|No. | Area | Observations(s) | Status | 
| :- | :- | :- | :- | 
|1 | Turbo - Revamp | Specifications for App Markup did not include the changes that should be implemented for Turbos - Revamp.  | Resolved |

### API
#### Accumulator
|No.  | Observations(s) | Status | 
| :- | :- | :- | 
|1 | App Markup is not reflected on contract.  | Resolved |
|2 | App Markup is not passed in sell causing POC to return an error.   | Resolved |
|3 | Take profit causing incorrect sell price.  | Resolved |
|4 | Setting take profit causing POC to be unable to be called.  | Resolved |

#### Turbo
|No.  | Observations(s) | Status | 
| :- | :- | :- | 
|1 | App Markup is not reflected on contract.  | Resolved |
|2 | App Markup is not passed in sell causing POC to return an error.   | Resolved |

#### Multiplier
|No.  | Observations(s) | Status | 
| :- | :- | :- | 
|1 | Stop Out barrier incorrectly calculated.  | Resolved |
|2 | Longcode shows incorrect multiplier amount.  | Resolved |
|3 | Deal Cancellation returns user input stake instead of final stake when exercised.  | Resolved |
|4 | Multiplier contract returned incorrect app_markup_amount.  | Resolved |
|5 | Take profit - Contract closed before take profit amount is reached.  | Resolved |

#### Vanilla
No observations found.

### Appendix
| Document | Link | 
| :- | :- | 
| Specifications | https://docs.google.com/document/d/1qVoByhGV8U36HGXhkRJ42hQiXY71SZYZdcXKgJU48Rw/edit?usp=sharing |
| Backend Specifications | https://docs.google.com/document/d/1v39fU8kc71OTJfT0H7iK6zHahyoZctryIM4pMuC-P3Y/edit?usp=sharing |

<br><br>
R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.
