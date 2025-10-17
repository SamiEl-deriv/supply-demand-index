# Stable Financial Spreads

Last updated: 2nd January 2025

### Squad Members

| Area             | Person In Charge                                    |
| :--------------- | :-------------------------------------------------- |
| Product Owner    | Chun Kiat (chunkiat@regentmarkets.com)   |
| Project Manager  | -|
| Backend          | Afshin Paydar (afshin.paydar@regentmarkets.com)  |
| QA               |     -       |
| Model Validation | Matthew (matthew.chan@regentmarkets.com)                    |

## Summary

The validation process for the Stable Financial Spreads model encompasses backend product specification and its implementation. This report summarizes validation findings, outlines the methodology used, and highlights key observations.





Total observations:

- 4 observations found.

Report in the form of slides [here](https://docs.google.com/presentation/d/1AKPni_RimRaaDeOXoBRtxY5KCtcCJbqCDLs3vdAI_qA/edit?usp=sharing). 
Working file for the validation process can be found in the same GitHub Directory as this report.

## Product information

The Stable Spread model aimed at providing more consistent and predictable spread for selected CFD instruments, utilizing raw feed data from our primary liquidity provider TopFX (with Bloomberg as backup). The initial phase covers 7 major FX pairs (EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD) and precious metals (XAU/USD, XAG/USD).

The model, powered by machine learning clustering algorithms, accounts for both seasonality patterns and economic event impacts on raw spreads. Starting with MT5 through a new designated trading group, this feature will provide traders with more reliable and stable spread conditions.



#### Underlying Symbols

- EUR/USD
- USD/JPY
- GBP/USD
- USD/CHF 
- AUD/USD 
- USD/CAD 
- NZD/USD
- XAU/USD
- XAG/USD



#### Offerings
- CFD

### Model Description
#### Seasonality Spread
- Using average spread of each timestamp (up to millisecond precision) within the lookback period.
- Apply K-mean clustering with 3 clusters
- 80th percentile oh each cluster is output as the spread value
- Obtain centroids of each cluster to then obtain threshold values
 
#### Economic Event Spread
- The fixed spread value for each regime, as well as their duration was calibrated with a statistical analysis of historical data around events.Those values depend on the ‘impact’ level of the event.
- The stable spread value was constructed around event using quantiles of the spread distributions. 80th quantile was proposed by quants for each window:
  - one before the event
  - one during the event  
  - one after the event



## Validation Methodology
### Overview
The validation process focused on comparing the model, as outlined in backend (BE) specifications, with the implementation code provided by the quants team. The process included:

- Verifying the soundness and completeness of BE specifications.
- Assessing code implementation against BE specifications.

### Validation Areas
- BE Specifications.
- Implementation Code (working files).

### Assumption
- It is assumed that the implementation files shared by the quants team represent the final implementation.


## Validation Findings

### Summary of Findings
Findings in slides format can be found [here](https://docs.google.com/presentation/d/1AKPni_RimRaaDeOXoBRtxY5KCtcCJbqCDLs3vdAI_qA/edit?usp=sharing)

| Scope                  | # Observations | Resolved | Non-blocker<br>(Future improvements) | Blocker |
| :--------------------| :------------: | :------: | :----------------------------------: | :-----: |
| <b>Total Observations |      <b>4     |    4     |                  0                  |    -    |
| BE Specification          |      1       |    1     |                  0                   |    -    |
| Implementation        |      3       |    3     |                  0                   |    -    |

## BE Specification


No. | Issue(s) | Status |
| :- | :- | :- |
|1 |  Different method of handling economic events spread configurations depending on how they overlap is not mentioned in BE specs. | Resolved |

## Implementation



No. | Issue(s) | Status |
| :- | :- | :- |
|1 | Feed data is resampled to 1 second intervals by using the first tick received, ignoring all other ticks received in that second. | Resvoled |
|2 | The handling of spreads for economic events is inconsistent based on how they overlap. | Resolved |
|3 | Event_spread based on BE spec should check +- 60 seconds before and after event timestamp, however code only checks +- 30 seconds. | Resolved |


