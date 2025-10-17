# Tests

Testing files for `deriv_quant_package`.

## `test_pricer`

A module containing the testing class `Test_Pricer` which tests the currently implemented pricers:

* Black-Scholes with the volatility smiles:
  * Flat
  * Spline (Quadratic/Cubic)
  * Implied Vanna-Volga
  * Vanna-Volga Approximated (1st & 2nd order)
* Vanna-Volga

### Implemented tests
The following tests implemented are to prevent various forms of arbitrage. Let:

* $C$: Call option price;
* $S$: Spot price;
* $K$: Strike price;
* $r$: Risk-free interest rate;
* $q$: Dividend rate;
* $t$: Time to expiry.

1. `MinStrikeTest`
  * The call option price $C$ cannot exceed $Se^{-qt}$ near $K=0$. In particular:
    $$
    \lim\limits_{K \rightarrow 0}C(K) = Se^{-qt}
    $$
2. `MaxStrikeTest`
  * $C$ must be positive at any point and should be near zero in the long run:
    $$
    \lim\limits_{K \rightarrow \infty}C(K) = 0
    $$
3.  `MinStrikeDerivativeTest`
  * The derivative $\frac{\partial C}{\partial K}$ must always exceed $e^{-rt}$ and should be near said value when $K$ is near zero:
    $$
    \lim\limits_{K \rightarrow 0}\frac{\partial C}{\partial K}(K) = -e^{-rt}
    $$
4. `MaxStrikeDerivativeTest`
  * The derivative $K\frac{\partial C}{\partial K}$ must never exceed 0 and should be near said value in the long run:
    $$
    \lim\limits_{K \rightarrow \infty}K \cdot \frac{\partial C}{\partial K}(K) = 0
    $$
5. `ConvexTest`
  * $C$ should be convex, that is, the second derivative $\frac{\partial^2 C}{\partial K^2}$ should be strictly positive at every $K$:
    $$
    \forall K, \enspace \frac{\partial^2 C}{\partial K^2} > 0
    $$
  * As a consequence, any narrow butterfly ($C(K+h) - 2C(K) + C(K+h)$ for small $h$) will not be negatively priced, hence preventing riskless profits.