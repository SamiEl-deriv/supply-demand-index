# Multiplier on Cryptocurrencies & Basket indices relaunch validation

Report Date - 25th June 2024

### <u>Squad Members</u>

| Area             | Person In Charge                                    |
| :--------------- | :-------------------------------------------------- |
| Squad Lead    | Yng Shan (yngshan@deriv.com)|
| Scrum Master     | Ekaterina Davidovich (ekaterina@deriv.com)|
| Backend          | Jun Bon (junbon@deriv.com)|
| QA               | Luqman (luqman@deriv.com) |
| Model Validation | Matthew (mathew.chan@deriv.com) |

## Summary

This report covers the validation on Multiplier next tick implementation on entry/start spot for DerivGO and API. The validation covers Cryptocurrencies and Basket Indices. Based on the validation, the next tick execution is now indeed implemented for both Cryptocurrencies and Basket Indices.

To summarise the results:

##### 1 Non-blocker:
The only point to note is that there is a possibility for the next tick to occur 2 seconds after the start time due to a missing tick in between. This is not an issue with the implementation of next tick execution.

## Context

A group of clients consistently profits from the Financial Multiplier product due to:

- The inability to hedge our product. 

- Latency by our current feed infrastructre.

The solution chosen is to implement the next tick as a simple solution. Details on the issue in [wiki js](https://wikijs.deriv.cloud/en/trading/dealing/trading-ops_non-linear/trading-incident-investigation-registry/incidents/incident_financial_multiplier_group_traders_2022).

## Validation Methodology

To check on DerivGO & API that all multiplier contracts on the listed underlyings are executed using next tick implementation.

### Assumptions

Testing was done in QA staging environment for both DerivGO and contracts bought using our API. 

#### Underlying Offerings Checked:

- Forex Basket Index
  - WLDAUD
  - WLDEUR
  - WLDGBP
  - WLDUSD
- Commodities Basket
  - WLDXAU
- Cryptocurrencies
  - cryBTCUSD
  - cryETHUSD

## Validation Findings

#### Summary of Findings


|No. | Observation | Status |
| :- | :- | :-|
|1 | There is a possibility for the next tick to be 2 seconds after start time due to missing tick. |  Non-Blocker |



#### Result 

|            Area             | Entry |
| :------------------------- | :--------: | 
|  FX Basket |   Next tick  | 
| Commodity Basket (Gold) |     Next tick |  
| Crypto |     Next tick        | 

Link to full results [here](https://docs.google.com/spreadsheets/d/1wa6dQaRuv4HCZ_4gVMRMyLaxg3vTT3E__360kErfmSU/edit?usp=sharing)



<br>R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

In line with the standards and criterias set out by the CEO, the Model Validation team has validated the product/indices as documented in this report.






