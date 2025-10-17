<h1 align="center"><b>Backend Index Specifications</b></h1> 

### Table of contents

1. [Introduction](#introduction)
2. [Index description](#index-description)
    1. [Content](#content)
    2. [General description](#general-description)
    3. [Algorithm description and formulas](#algorithm-description-and-formulas)
3. [Index parameters](#index-parameters)
4. [Spread](#spread) 
    1. [Formulas](#formulas)
    2. [BO parameters](#bo-parameters)
    3. [BE and BO tables](#be-and-bo-tables) 
    4. [Requirements](#requirements)
5. [Platforms](#platforms)
6. [Trading conditions](#trading-conditions)
7. [Cross referencees](#cross-references)
8. [Specificities](#specificities)
9. [Acronyms](#acronyms)
10. [Annex](#annex)
    1. [Geometric brownian motion](#geometric-brownian-motion)
    2. [Index stochastic differential equation solution](#index-stochastic-differential-equation-solution)
    3. [Drift correction term](#drift-correction-term)
    4. [Inference algorithm](#inference-algorithm)
    5. [Diff indicator](#diff-indicator)
    6. [Spread construction](#spread-construction)


## Introduction

<!-- None of the synthetic indices allowed clients to trade different clear trends within a day. It is from this observation that the **Drift Switch Index** idea was born.

The idea behind the drift switching index is to create a new attractive index for clients which does not depend only on random movement. It will be proposed to customers as a 
volatility index with clear random trends appearing randomly. It would lead customers to take advantage of certain trends by making assumptions about the possible trend (regime).  -->

<!-- In that way, the Drift Switch Index have been designed. 3 Drift Switch Index are launched under the names of DSI10, DSI20 and DSI30. -->

</br> </br>


## Index description

### Content 

<!-- The Drift Switch Index is a synthetic index (DSI) created to replicate real market world scenarios. Financial markets are affected by the economic cycles. It is noticeable 
with long term trends. This index mimics this idea of bullish and bearish trends over time.  

The idea is to make happen random trends during the lifetime of the index : 

- Positive Drift Regime (also known as a Bullish Trend),
- Negative Drift Regime (also known as a Bearish Trend), 
- Driftless Regime (also known as a Sideways Trend)

ThIs index will enable clients to make assumptions on the current regime. Clients will take trades based on their assumptions.

Clear trends will appear randomly during a certain amount of time. From a client’s perspective, this index would be attractive since during an observed upward or 
downward price trend, the client is led to believe the market is currently in a drift regime, and hence that the price is likely to continue to trend. -->

</br>

### General description

The Volatility Switch Index will evolve according a volatility index. In order to make appear some regime, random  will be choosen for  the model. 

The index will be simulated following Geometric Brownian Motion (GBM). The GBM is a continuous-time stochastic process in which the logarithm of the random part varies according 
a Brownian motion (also called Wiener process). Find in annex [Geometric Brownian Motion](#geometric-brownian-motion) some justifications about this mathematical model. 

In addition, the index handle the state of parameters inside the model. It has been decided to control the vol term using Markov chain. It means that the vol of the process is not constant over the time. However the term of trend (drift) is constant over the time. 

This can be considered has adding multiple regime in the model. An analogy can be made with the different economic cycles visible in the financial markets. Different regimes characterised the index, and their specificities are explained below :


<!-- |  | **No trend**  | **Bull market** | **Bear Market** | 
| :---: | :---: | :---: | :---: |  
| **Regime designation** | Stationary regime | Positive regime | Negative regime  | 
| **Features** | GBM +0 drift | GBM with + $\mu$ drift | GBM with -  $\mu$ drift | 
 -->

</br>

 ### Algorithm description and formulas

According to the previous statements, the Drift Switch Index can be modeled using a random variable.

Let $(\Omega, F, (F_{t})_ {t \geq 0}, P)$ be a probability space satisfying that $(F_t)_ {t \geq 0}$ is a right continuous filtration for each $t \geq 0$. Let's consider $(X_t)_ {t \in R_*^+}$ the value of this index at every time $t \in [0, + \infty[$. The equation satisfied by this random variable come as :

$$
\begin{equation}
    \frac{dX_t}{X_t} = \mu dt + \sigma(\alpha_t)  dW_t, \ \ X_0 = x \\
\end{equation}
$$


- $X_t$ = Drift Switch Index price
- $\mu$ = a conatant drift 
- $\alpha_t$ = a continuous-time Markov chain which represent the regime
- $\sigma(\alpha_t)$ = a volatility function
- $W_t$ = standard Brownian motion
- x = initial index value

</br>

As mentioned in the section above, the vol $\sigma (\alpha_t)$,  $t \in {R}$ is not constant. It has been decided to model the vol using Markov chain. So the vol may take N different values depending on the a Markov chain $(\alpha_t)_ {t \in R_*^+}$ value. More specifically, the volatility function can be written as 

$$
\sigma(\alpha_t = i) = \sigma_i 
$$

</br>

$(\alpha_t)_{t \in {R}^+}$ represents the regime at time t. It has been assumed that $(\alpha_t)_{t \in {R}^+}$ and $(W_t)_ {t \in R_*^+}$ are two independent process. It can be equal to 3 different values which determine the drift regime. The index equation becomes : 

$$
\begin{equation}
    dX_t = X_t \left( \sum_{i=0}^2 \mu_i {1}_{\alpha_t = i} dt + \sigma_i dW_t \right), \ \ X_0 = x
\end{equation}
$$

</br>

Each of the $N$ regime will occur according to a certain probability. It means that at time t, there is a specific probability ${P}_{\lambda}$ to switch from regime i to regime j. Let's denote this probability ${p_{ij}}$. Depending on the two configurations there is different probability to switch from drifted regime to another one. The 2 ways model doesn't allowed to move from drifted regime from the other drifted regime.

The probability to switch from a regime to another can be summarize using stochastic matrix. This matrix is a square matrix used to describe the transitions of a Markov chain. This matrix will be denoted P and comes as follows :

$$
\begin{equation*} 
P = 
    \begin{pmatrix} 
        p_{00} & p_{01} & ... & p_{Nj} \\ 
        p_{10} & p_{11} &  ... & p_{1N}\\
        : & : & ... & :\\
        p_{N0} & p_{N1} & ... & p_{NN} \\
    \end{pmatrix}
\end{equation*}
$$

$$
\begin{equation}
    p_{ij} = {P} \left( \alpha_t = i | \alpha_{t-1} = j \right) \ \ \ with \ \ 0 \leq i,j \leq 2 \ \ \ \ \ \ \ and \ \ \sum_{i=0}^N p_{ij} = 1
\end{equation}
$$

</br>

It means that $p_{0, 1}$ is the probability of going from the the regime 0 (positive regime - $\mu_0 = + \mu$) to the regime 1 (stationary regime - $\mu_1 = 0$). This probability is equal to $\frac{\lambda}{2}$ according to the transition probability matrix. Below is the transition matrix associated to the two different DSI configurations : 


$$
\begin{equation*} 
P = \frac{1}{N-1}
    \begin{pmatrix} 
        1 - (N-1)\lambda_0 &  \lambda_1& ...  & ... & \lambda_N \\ 
        \lambda_0 & 1 - (N-1)\lambda_1 & \lambda _2 &... & \lambda_N \\ 
        : & : & : & : & : & \\
        \lambda_0 & \lambda_1 & ... &\lambda_{N-1 }& 1 - (N-1)\lambda_N \\
    \end{pmatrix}
\end{equation*}
$$

</br>



The parameter $\lambda_i$ is the inverse of $T_i$ where $T_i$ is the **characteristic time of a regime**. It represents the average duration of a regime. For example, if the stationary regime last in average 30 minutes, this characteristic time is expressed has follows:

$$
\begin{equation*}
    \lambda_i = \frac{1}{T_i} = \frac{1}{time \ in \ seconds} = \frac{1}{30 \times 60}
\end{equation*}
$$

</br>

<!-- Initial probabilities are defined as $\xi = \left[ P(\alpha_0 = 0), \ P(\alpha_0 = 1), \ P(\alpha_0 = 2) \right]$. The initial vector is set up as [0, 1, 0] in the tick engine. -->

</br>

## Index parameters

</br>

<!-- |  | **Volatility - $\sigma$**  | **Drift - $\mu$** | **Regime avg duration - $T$** | **Gamma - $\gamma$** | **Initial spot** | 
| :---: | :---: | :---: | :---: |  :---: | :---: | 
| **DSI10** | 10% | 100 | 10 min  | 0.4980997907 | 10000 |
| **DSI20** | 10% | 60 | 20 min | 0.4977183219 | 10000 |
| **DSI30** | 10% | 35 | 30 min | 0.4980031156 | 10000 |

</br>

|  | **Implied volatility**  | 
| :---: | :---: | 
| **DSI10** | 41% |
| **DSI20** | 35% | 
| **DSI30** | 27% | 
 -->

</br> </br>

## Spread
<!-- 
The spread model for the Volatility Switch Index has been determined such that it will replicate financial market spreads. This spread has been divided into two : $Spread_{ask}$ ($S_{ask}$) and $Spread_{bid}$ ($S_{bid}$). In that sense 3 configurations can be considered : 


- Clear bullish trend : large buying period so less liquidity on buy side which means higher $S_{ask}$. 
- Clear bearish trend : large selling period so less liquidity on sell side which means higher $S_{bid}$. 
- No clear trend : $S_{ask}$ and $S_{bid}$ will be close and pretty small. 

As it is describe, the spread is not constant. The optimal spread has been determined using inference algorithm. This algorithm is described in annex. -->

</br>

### Formulas 

The spread is divided in two parts : 

- Fair spread percentage, that gives the part we should charge to not to be exploited.
- Commission, that gives the commission for such an index.

</br>

Here is the full formula for the $Ask$ and $Bid$ price:

$$
\begin{equation}             
Ask(x)=Spot \times (1+FairSpread(x)/2)+CommissionAsk(x)
\end{equation}
$$

$$
\begin{equation}             
Bid(x)=Spot \times (1-FairSpread(x/2))+CommissionBid(x)
\end{equation}
$$

</br>

<!-- The variable x refer to the diff indicator ($diff$). This indicator results from the inference algorithm and represents the difference of probability to be in positive and negative regime. This indicator is discribe in the annex described in the annex ([Diff indicator](#diff-indicator)). Such a parameter is updated every seconds and store in Deriv database. -->

</br>

**Fair spread percentage**
</br>

Fair spread are a ponderation of faire spread defined on other vol index.

$$
\begin{equation}             
FairSpread = \xi_t . S_\sigma
\end{equation}
$$

where $S_\sigma$ is a $(N\times 1)$ vector with ${S_\sigma}_i  = S_t * \sigma_i$ which is the typical spread for a vol index of volatility $\sigma_i$.
$\xi_t$ is the current state vector,${\xi_t}_i$ is the proba to be in state $i$ at time $t$. It is estimated using the same inference algorithme that for DSI.


<ins> **Commission** </ins>

<!-- $$
\begin{equation}             
CommissionAsk(x)=\frac{pip}{2} max((com_{Max}-com_{Min}) \times x +com_{Min} , 0) 
\end{equation}
$$

$$
\begin{equation}             
CommissionBid(x)=\frac{pip}{2} max((com_{Min}-com_{Max}) \times x +com_{Min} , 0) 
\end{equation}
$$ -->

</br>

### BO parameters 
<!-- 
</br>
<!-- 
|  | **Tick size**  | **Base currency** | **Contract size** | **Gen interval** | 
| :---: | :---: | :---: | :---: |  :---: | 
| **Parameters** | 0.01 | USD | 1 |  1 second |

</br>

A back office tool has been deployed in order to manage spread parameters. This tool is called BO tool. The following parameters can be changed easily has the will of the dealing team : 

- Perf : The performance of the clients. 100% is the expected performance but it could be increased/lowered if they over/under-perform (based on expected revenue calculation)

- $com_{Min}$ : the minimum we charge as commission (constitutive of our revenue) in number of pips

- $com_{Max}$ : the maximum we charge as commission (constitutive our revenue) in number of pips
 --> -->
<!-- 
</br>

### BE and BO tables 


<ins> **BE table** </ins>

<table>
    <thead>
        <th>
            <th colspan=4>Hard coded (BE)</th>
        </th>
    </thead>
    <tbody>
        <tr>
            <th></th>
            <th colspan=2> Spread Ask</th>
            <th colspan=2> Spread Bid</th>
        </tr>
        <tr>
            <th></th>
            <th>Kappa ref</th>
            <th>Epsilon ref</th>
            <th>Kappa ref</th>
            <th>Epsilon ref</th>
        </tr>
        <tr>
            <th>DSI10</th>
            <th>0.1268393717%</th>
            <th>0.0001608149%</th>
            <th>0.12635732896%</th>
            <th>0.00032122775%</th>
        </tr>
        </tr>
            <th>DSI20</th>
            <th>0.152207353844%</th>
            <th>0.000231621965%</th>
            <th>0.151512777478%</th>
            <th>0.000462954401%</th>
        </tr>
        </tr>
            <th>DSI30</th>
            <th>0.1331813546891%</th>
            <th>0.0001773476251%</th>
            <th>0.1326494599%</th>
            <th>0.0003545474%</th>
        </tr>
    </tbody>
</table>

</br>

<ins> **BO tables** </ins>

<table>
    <thead>
        <th>
            <th colspan=3>BO</th>
        </th>
    </thead>
    <tbody>
        <th>
            <th colspan=3> Indicative value</th>
        </th>
        <tr>
            <th></th>
            <th>Indicative ComMin</th>
            <th>Inidcative ComMax</th>
            <th>Indicative Perf</th>
        </tr>
        </tr>
            <th>DSI10</th>
            <th>73 pips</th>
            <th>146 pips</th>
            <th>100%</th>
        </tr>
        </tr>
            <th>DSI20</th>
            <th>62 pips</th>
            <th>125 pips</th>
            <th>100%</th>
        </tr>
        </tr>
            <th>DSI30</th>
            <th>48 pips</th>
            <th>96 pips</th>
            <th>100%</th>
        </tr>
    </tbody>
</table>

</br>

<ins> **Constraints** </ins>

<table>
    <thead>
        <th>
            <th colspan=4>Minimum</th>
        </th>
    </thead>
    <tbody>
        <th>
            <th>Com Min</th>
            <th>Com Max</th>
            <th>Perf</th>
        </th>
        </tr>
            <th>DSI10</th>
            <th>0 pips</th>
            <th>0 pips</th>
            <th>0 pips</th>
        </tr>
        </tr>
            <th>DSI20</th>
            <th>0 pips</th>
            <th>0 pips</th>
            <th>0 pips</th>
        </tr>
        </tr>
            <th>DSI30</th>
            <th>0 pips</th>
            <th>0 pips</th>
            <th>0 pips</th>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <th>
            <th colspan=4>Maximum</th>
        </th>
    </thead>
    <tbody>
        <th>
            <th>Com Min</th>
            <th>Com Max</th>
            <th>Perf</th>
        </th>
        </tr>
            <th>DSI10</th>
            <th>Spot / pip . 0.1270% . pips</th>
            <th>Spot / pip . 0.1270% . pips</th>
            <th>200%</th>
        </tr>
        </tr>
            <th>DSI20</th>
            <th>Spot / pip . 0.1524% . pips</th>
            <th>Spot / pip . 0.1524% . pips</th>
            <th>200%</th>
        </tr>
        </tr>
            <th>DSI30</th>
            <th>Spot / pip . 0.1334% . pips</th>
            <th>Spot / pip . 0.1334% . pips</th>
            <th>2005</th>
        </tr>
    </tbody>
</table>

</br> -->
<!-- 
As the Spot is actually the Bid on MT5, we can use the Bid value on previous calculations.

The reason for setting these minimum values is quite straightforward. For maximum values:

- For $Com_{Min}$ and $Com_{Max}$ , we consider that we will never charge more than the fair spread
- For $Perf$, we consider that if clients perform more than 200% of what we expected, there is a problem in the index and we should remove it from the proposed indices

</br> -->

### Requirements
<!-- 
All the parameters can be changed in the BO tool. Here is the requirement for the parameters :

- $Com_{Min}$, $Com_{Max}$ and $Perf$ Indicative: are just indicative values. Moreover the indicative values for $Com_{Min}$ and $Com_{Max}$ are based on calculations that are done when the spot of the market is roughly $10k

- Pip value is $0.01, which means for example that if we want to charge $1 as minimum commission, we should enter 100 pips for ComMin in BO

Caution : Values of $Com_{Min}$, $Com_{Max}$ and $Perf$ should be updated regularly as:

- $Perf$ reflects the global performance of the users, which changes when globally users change their strategy. Thus, in order to know if clients are over/under-performing, keep tracking regularly the realized revenue and compare it to the expected revenue.

- $Com_{Min}$, $Com_{Max}$ are expressed in pips, thus, if the market grows to two times the original spot it means that we charge two times less than what we previously charged (and the converse is also valid in the converse direction). Be careful, track the evolution of the market and make $Com_{Min}$, $Com_{Max}$ consistent with it.

</br> </br> -->

## Platforms 
<!-- 
The Drift Switch Index is available on MT5, Deriv X and Deriv EZ. It will be soon available on CTrader

</br> -->

## Trading conditions 

<!-- | **Symbols**  | **Base currency** | **Contract size** | **Tick size** | **Leverage** | 
| :---: | :---: | :---: | :---: | :---: | 
| DSI10 - DSI20 - DSI30 | USD | 1 | 0.01 | 500 |  

</br> -->

### MT5 Swap rate
<!-- 
Swap fee is charged when a client holds the position overnight. Let’s define the swap rate as follow :


$$
\begin{equation*}
    swap \ rate = realised \ volatility \times 10
\end{equation*}
$$

</br>

Considering the DSI30 which have a realised volatility of 41 %, the amount a client will be charged is:

$$
\begin{equation*}
    swap \ fee \ in \ dollar = volume \ in \ lot × current \ price × \frac{Swap \ rate}{365}
\end{equation*}
$$

</br>

| **Swap rate** | **Swap rate per lot (in $)**|
| :---: | :---: |
| 4.1% | 1.123|
 -->

</br>

<!-- ### Min - Max lot size

Min lot size is defined to have IB’s commission at least 0.01\$ for min lot.

Max lot size is set to make volatility-adjusted USD Volume approximately equal across all Synthetic assets.

| **Min lot size** | **Max lot size**|
| :---: | :---: |
| 0.01 | 10 | -->

</br>

### IB Commission
<!-- 
The IB commission is set up in order to offer 40% of the spread as commission. Since the spread is not constant over the time, it has been chosen to take 40% of the median of the spread. The distribution of the spread gives the median as 0.00060. It means that the IB commission will be 0.000238.  -->

</br>

### Leverage and margin

<!-- The leverage for Drift Switch Index is 500 for thr Rest Of the World (ROW).

| **Initial Margin Rate** | **Margin required (in USD)**|
| :---: | :---: |
| 0.50 | 10 | -->

</br> </br>

## Cross references

### Relevant links
<!-- 
[DSI - BO specs](https://docs.google.com/document/d/13lLR7P7x-nPd4F37sjsl8GGSPBM4vm_RtD0XRXBnZqE/edit?usp=sharing)

[DSI - BE specs](https://docs.google.com/document/d/1YAQsODIAgNsGAeVUwAKVsDImx4riL5xTdQA9NAegwBE/edit?usp=sharing)
 
[DSI - BO Tool](https://docs.google.com/spreadsheets/d/10GROML2A0XabtyjF0D7qPcX_XKmD0nv5_0VGts60lto/edit?usp=sharing)

[DSI - BO Tool explained](https://docs.google.com/document/d/1Xm5IP_ewjQCwKKLPqJTz7DuiI4xzI3v29m4OJPmhHj4/edit?usp=sharing)

</br> -->

### Code links 
(To be updated)

[Python code]()

[Perl]([https://github.com/afshin-deriv/perl-Feed-Index-DriftSwitch/tree/feed/create_index-drift-switch](https://github.com/regentmarkets/perl-Feed-Index-DriftSwitch))

[Quant respository](https://github.com/afshin-deriv/perl-Feed-Index-DriftSwitch/tree/feed/create_index-drift-switch)


</br> </br>

## Specificities

<!-- This section gives the specificities of the index, not only its construction but also about its implementation. 

- The spread is not constant as other synthetic index. It changes according to the probability to be in each regime. The spread model is justified in section [Spread](#spread).
- As the spread is not constant, the Backend team deployed a tool called BO Tool which enable to change spread parameters without restarting the feed generation. 
- The spot price cannot be displayed by financial platforms (MT5, Deriv X...). That's why the sell price is displayed on each platform the DSI is offered.
- About once a week the tick engine need to be restart. Such the state of the DSI cannot be store in Deriv DB, it has been decided to generate randomly the state of the DSI every time the engine is restarted.

</br> </br> -->


## Acronyms 

<!-- - **BS** : Black \& Scholes

- **BO** : Back-office 

- **CFD** : Contract for difference 

- **com** : Commission

- **DSI** : Drift Switch Index

- **MT5** : MetaTrader 5

- **SDE** : Stochastic Differential Equation


</br> </br> -->


## Annex

### Geometric Brownian Motion

<!-- A stochastic process $X_t$ is said to follow a GBM if it satisfies the following stochastic differential equation (SDE):

$$
\begin{equation}
     dX_t =\mu X_t dt + \sigma X_t  dW_t 
\end{equation}
$$

where $W_t$ is a Wiener process or Brownian motion, and $\mu$ the drift term and $\sigma$  diffusion term (also called volatility) are constants. 

</br> -->

------

<!-- <ins> **Ito lemma** </ins>

Suppose $f : {R} \rightarrow {R}$ is twice continuously differentiable and $dX_t = \alpha_t dt + \beta_t dW_t$. Then f(X) is the Ito process,

$$
\begin{equation*}
    f(X_t) = f(X_0) +  \int_0^t f(X_s) \alpha_s ds +  \int_0^t  f'(X_s) \beta_s dW_s + \frac{1}{2} \int_0^t f''(X_s) \beta^2_s ds \ \ \ for \ t \geq 0
\end{equation*}
$$

In differential form, Ito’s lemma becomes :

$$
\begin{equation*}
    df(X) = f(X) \alpha_t dt + f(X) \beta_t dW_t + \frac{1}{2} f(X) \beta_t^2 dt.
\end{equation*}
$$

---- -->

</br>

<!-- Using Ito lemma, this equation becomes : 

$$
\begin{equation}
     X_t = X_0 e^{ \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t}
\end{equation}
$$


</br> -->

<!-- ### Index Stochastic Differential Equation solution

The Drift Switch Index is modeled according to SDE below. Justification of terms can be seen in section Algorithm. 

$$
\begin{equation}
    \frac{dX_t}{X_t} = \mu_t( \alpha_t) dt + \sigma dW_t, \ \ X_0 = x
\end{equation}
$$

Using differential form of Ito's lemma, it comes :

$$
\begin{align*}
    d(\ln X_{t}) &= (\ln X_{t})'dX_{t}+\frac{1}{2}(\ln X_{t})''dX_{t}dX_{t} \\
                 &= \frac {dX_{t}}{X_{t}} - \frac{1}{2X_{t}^{2}} \sigma^2 X_t^2 dt \\ 
                 &= \mu_t( \alpha_t) dt + \sigma dW_t - \frac{1}{2} \sigma^2 dt \\
                 &= \left(\mu_t( \alpha_t) - \frac{1}{2} \sigma^2 \right)dt + \sigma^2 dW_t 
\end{align*}
$$

Then by integrating it comes :

$$
\begin{align*}
    \ln X_{t} &= \int_0^t \left(\mu_s( \alpha_s) - \frac{1}{2} \sigma^2 \right)ds + \int_0^t \sigma^2 dW_s \\
              &= \int_0^t \mu_s( \alpha_s) ds - \frac{1}{2} \sigma^2 t + \sigma^2  W_t
\end{align*}
$$

Using the exponential as the reciprocal bijection of the neperian logarithm function : 

$$
\begin{equation*}
     X_t = X_0 e^{ \int_0^t \mu_s ds - \frac{\sigma^2}{2} t + \sigma W_t} \ \ \ with \ X_0=x
\end{equation*}
$$

$$
\begin{equation*}
     X_t = x e^{ \int_0^{t} \sum_ {i=0}^2 \mu_i {1}_{ \alpha_s = i} ds - \frac{\sigma^2}{2} t + \sigma W_t} 
\end{equation*} 
$$

</br> -->


### Inference algorithm

The process $(X_t)_ {t \in R_*^+}$  which describe the index is latent. It means that client will never know for sure which regime prevailed at a certain point in time. However, to make assumptions on current regime, the information from the current and past observations, combined with the distributions and transition probabilities can be used to make inference.
 <!-- Considering that the algorithm knows all parameters from the model, it computes probabilities to be in each regime. For example, at time t the model will give : 

- ${P}$ (0 drift regime) = 0.12 
- ${P}$ (positive drifted regime) = ${P}_{+\mu}$  = 0.68 
- ${P}$ (negative drifted regime) =  ${P}_{-\mu}$ = 0.20  -->

 Those probability can be estimate using an inference model. In fact, it uses probabilities concepts to compute at every time, the probability to be in each regime. 
 $$
 \xi_t = \left[ {P} (\alpha_t = 0), {P} (\alpha_t = 1), {P} (\alpha_t = 2),...\right]
 $$


As a reminder the process $(X_t)_ {t \in R_*^+}$  which represents the index evolution is modeled according a geometric Brownian motion using Markov chain for volatility modeling. Then the returns, denoted by $R_t$ follows a distribution that depends on the process $\alpha_t$ : 

$$
\begin{equation}
R_t \sim
    \begin{cases}
        \mathcal{N}(- \frac{\sigma_0^2}{2},\,\sigma_0^{2}) \ \ &if  \ \alpha_t = 0 \\ \\
        : \\ \\
        \mathcal{N}(-\frac{\sigma_i^2}{2},\,\sigma_i^{2}) \ \ &if \ \alpha_t = i \\ \\ 
        :
        \\ \\
        \mathcal{N}( - \frac{\sigma_N^2}{2},\,\sigma_N^{2}) \ \ &if  \ \alpha_t = N \\ 
    \end{cases}       
\end{equation}
$$


Using the past returns observations combined with distributions and transition probabilities  ${P} \left( \alpha_t = i | r_t, r_{t-1},...,r_1 \right)$ can be estimated. Using Bayes' rule \ref{Bayes}, and the Law of Total Probability, it follows  that : 

$$
\begin{align*}
{\xi_{t|t-1}}_i &={P} ( \alpha_{t} = i| R_{t-1} = r_{t-1} ) \\
& = \sum_j{P} ( \alpha_{t} = i| \alpha _{t-1}=j)*P(\alpha_{t-1}=j|R_{t-1} = r_{t-1})
\end{align*}
$$
which give  = ${\xi_{t|t-1}} = P.\xi_{t-1|t-1}$
and $P.\xi_{t-1|t-1}$ can be compute as follow
$$
\begin{align*}
    {P} ( \alpha_{t-1} = j| R_{t-1} = r_{t-1} ) &= \frac{{P} ( R_{t-1}  = r_{t-1}| \alpha_{t-1} = j ) \times {P} ( \alpha_{t-1} = j ) } {{P} ( R_{t-1} = r_{t-1} )} \\
    r_{t-1} ) &= \frac{{P} ( R_{t-1}  = r_{t-1}| \alpha_{t-1} = j ) \times {P} ( \alpha_{t-1} = j ) } {\sum_i {P} ( R_{t-1}  = r_{t-1}| \alpha_{t-1} = i ) \times {P} ( \alpha_{t-1} = i )}
\end{align*}
$$


Let's denote $\xi_{1|0} = \left[ {P} (\alpha_0 = 0), {P} (\alpha_0 = 1), {P} (\alpha_1 = i), ...\right] = (1, 0,..., 0)$. Then the next regime probability can be determined using the transition probability matrix P.

$$
\begin{equation*}
f_t = 
    \begin{pmatrix}
        f(r_t; 0, \sigma_0^2) \\ \\
        :\\ \\
        f(r_t; 0, \sigma_i^2) \\ \\
        :\\ \\
        f(r_t; 0, \sigma_N^2) 
    \end{pmatrix}
\end{equation*}
$$

The algorithm scheme then becomes : 

$$
\begin{align}
    \xi_{t|t} = \frac{1}{\xi_{t|t-1}f_t} \xi_{t|t-1} \odot f_t 
\end{align}
$$

$$
\begin{align}
    \xi_{t|t-1} = P \frac{1}{\xi_{t-1|t-2}f_t(r_{t-1})} \xi_{t-1|t-2} \odot f_t (r_{t-1})
\end{align}
$$

where $\odot$ equals element-wise multiplication. Thus, knowing all the information and the past realisation, the probabilities to be in each regime can be easily known. Those probabilities will be used to set up the diff indicator.

</br>

### Diff indicator
<!-- 
What is called Diff indicator for difference indicator ? The diff indicator is a probability differential indicator which results from the inference algorithm. It is computed according the following formula :

$$
\begin{equation}
    diff = {P}_ {+\mu} - {P}_ {-\mu} 
\end{equation}
$$

This indicator will gives us more information about the current regime. In fact, it gives the regime that that is the most likely. Thus, two configurations are possible : 

- ${P}_ {+\mu} > {P}_ {-\mu} \Rightarrow diff > 0$ 
- ${P}_ {-\mu} > {P}_ {+\mu} \Rightarrow diff < 0$

The diff indicator moves between -1 and 1 : 

- Diff indicator close to -1 mean that the current regime is almost surely a negative regime, 
- Diff indicator close to 1 mean that the current regime is almost surely a positive regime.

</br> --> -->

### Spread construction

In order to construct the spread, a strategy as been used. The perfect strategy will assume, knowing at anytime the states of the DSI, to take long and short position according to the corresponding state. This strategy will lead to a specific expected return ($e^{\mu T dt} - 1 \sim \mu T dt$).

Since the state is not known at anytime, a more reasonable strategy has been used. Using the diff indicator (above described) denoted x, a signal can be defined : 

- Long if x is positive
- Short if x is negative 
- Close if x switches sides

Then the expected PnL here is $\mu dt T |x|$. The spread has been based on this results. In that sens this expected PnL should be the PnL client can get with the information they get. 









R&D effort needs to be in line with Deriv’s vision and mission as formulated by our CEO. Therefore all R&D projects are carefully selected by our C-Level senior management represented by JY and Rakshit and resources for the projects are only allocated after review and shortlisting based on their vision and priorities. 

The first stage of the Working-Backwards process includes PoC, Products Specs, mock design and Key Important Document (KID) to get feedback from the design team and approval from Compliance before we start committing resources and efforts into the implementation.
