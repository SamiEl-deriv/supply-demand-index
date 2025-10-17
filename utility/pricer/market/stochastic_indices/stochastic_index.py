from typing import Type
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm, foldnorm, bernoulli, poisson, expon,\
      rv_continuous
from typing import Union
from .stochastic_index_base import StochasticIndex, geometricIndexDecorator, dt
from scipy.signal import fftconvolve


@geometricIndexDecorator
class VolIndex(StochasticIndex):
    """
    Volatility index class.

    This is the implementation of a Black-scholes model index with
    a constant volatility.
    """

    def __init__(
            self,
            volatility: float,
            yield_rate: float = 0.0) -> None:
        """
        Parameters
        ----------
        volatility : float
            The volatility of the index expressed in terms of percentage
            per square root of years.
        interest_rate : float
            The interest rate associated to the index (in terms of percentage
            per years). Default is 0.
        dividend_rate : float
            The dividend yield associated to the index (in terms of percentage
            per years). Default is 0.
        """
        self.__yield_rate = yield_rate
        self.__volatility = volatility

    def _drawMarginalReturn(self, dt: float = dt,
                            size: Union[int, tuple[int]] = 1,
                            seed: int = None) -> np.ndarray:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """
        return self._margiReturnDistrib(dt=dt).rvs(size=size,
                                                   random_state=seed)

    def _margiReturnDistrib(self, dt: float = dt) -> np.ndarray:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """
        return norm(loc=self.drift*dt,
                    scale=self.__volatility*np.sqrt(dt))

    def _cumulReturnDistrib(self, T: float,
                            returnsInit: Union[float, np.ndarray] = 0.0
                            ) -> rv_continuous:
        """
        Final cumulated returns distribution

        Parameters
        -------
        T        : float
            Time length of the stochastic process
        Returns
        -------
        ss.rv_continuous
            The final cumulated returns distribution at time t
        """
        return norm(loc=self.drift*T + returnsInit,
                    scale=self.volatility*np.sqrt(T))

    def _condiReturnDistrib(self, T: float,
                            return1: Union[float, np.ndarray], T1: float,
                            return2: Union[float, np.ndarray], T2: float
                            ) -> rv_continuous:
        mean = ((T2-T)*return1 + (T-T1)*return2) / (T2-T1)
        var = self.__volatility**2 * (T2-T) * (T-T1) / (T2-T1)
        return norm(loc=mean, scale=np.sqrt(var))

    def expectedFwdValue(self, S0: float, T: float) -> float:
        """
        Method that return the expected forward value of the index

        Parameters
        ----------
        S0 : float
            The initial spot value of the index.
        T   : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index
        """
        return S0 * np.exp(- self.drift * T)

    @property
    def yield_rate(self) -> float:
        """
        Retrieve interest rates

        Returns
        -------
        float
            The interest rate
        """
        return self.__yield_rate

    @property
    def volatility(self) -> float:
        """
        Retrieve volatility

        Returns
        -------
        float
            The volatility
        """
        return self.__volatility

    @property
    def drift(self) -> float:
        """
        Drift of the asset price

        Returns
        -------
        float
            interest_rate - dividend_rate
        """
        return self.__yield_rate - self.__volatility**2/2

# TODO: Once new RPG module works fine, backward_bridging would be depreciated
    def backward_bridging(self, S_0: float,
                          final_creturns: np.ndarray,
                          T: float,
                          dt: float = dt,
                          seed: int = None) -> np.ndarray:
        """
        Method that constructs paths of cumulated returns conditionnally on 
        the final cumulated returns.

        Parameters
        ----------
        S_0            : float
            The initial index spot value
        final_creturns : np.ndarray
            The final cumulated returns array
        T              : float
            Final time of the path
        dt             : float
            Timelapse between two steps in the stochastic process.
            By default equal to 1s (in years)
        seed           : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
            An array of float where each lines corresponds to a different
            path and each columns to a different timestep of the stochastic
            process followed by the cumulated returns starting fom 0 and
            finishing to values given in 'final_creturns'.
        """
        N_t = int(T/dt)
        shape = (final_creturns.shape[0], N_t+1)
        cumreturns = np.empty(shape)
        cumreturns[:, 0], cumreturns[:, -1] = 0, final_creturns
        rand_norm = norm.rvs(size=shape, random_state=seed)
        for n in range(N_t-1, 0, -1):
            cumreturns[:, n] = n/(n+1)*cumreturns[:, n+1] - self.drift*dt +\
                  self.__volatility * np.sqrt(n/(n+1)*dt)*rand_norm[:, n-1]
        return S_0 * np.exp(cumreturns)


@geometricIndexDecorator
class CrashIndex(StochasticIndex):
    """
    Crash index class
    """

    # The mean of the |N(1,1)| distribution
    global mu
    mu = np.sqrt(2 / np.pi) * np.exp(-1 / 2) + 1 - 2 * norm.cdf(-1)

    def __init__(self, index: int) -> None:
        """
        Parameters
        ----------
        index : int
            The Id number of the Crash index that represent the average
            number of second between two jumps (ex: 1000 for 'CRASH1000')
        """
        self.__Pu, self.__Pd, self.__MUT, self.__MDT,\
            self.__volatility = CrashIndex.get_params(index)

    @staticmethod
    def get_params(index: int) -> tuple[float, float, float, float, float]:
        """
        Static method that returns the properties of the Crash/Boom index

        Parameters
        ----------
        index : int
            The Id number of the Crash index that represent the average
            number of second between two jumps (ex: 1000 for 'CRASH1000')

        Returns
        -------
        tuple of floats
            The probability of spot moving up, the probability of moving
            down, the mean up tick, mean down tick and the corresponding
            volatility for a vol index (for large timescale, Crash/Boom
            indices are equivalent to a vol index)
        """

        P_d = 1 / index
        P_u = 1 - P_d
        MDT = np.log(P_u) / np.sqrt(dt)
        t2 = MDT * dt / mu
        def factor(x): return np.exp(x**2 / 2 + x) * \
            norm.cdf(1 + x) + np.exp(x**2 / 2 - x) * norm.cdf(x - 1)

        def f(x): return 1 - factor(t2) * P_d - factor(x) * P_u
        z = fsolve(f, 0)[0]
        MUT = z / dt * mu
        VOL = np.sqrt(2 / mu**2 * (P_u * MUT**2 + P_d * MDT**2 +
                      4 * P_d * P_u * MUT * MDT) - (P_u * MUT + P_d * MDT)**2)
        return np.array([P_u, P_d, MUT, MDT, VOL])

    @property
    def up_prob(self) -> float:
        """
        Retrieve the moving up probability

        Returns
        -------
        float
            The moving up probability
        """
        return self.__Pu

    @property
    def down_prob(self) -> float:
        """
        Retrieve the moving down probability

        Returns
        -------
        float
            The moving down probability
        """
        return self.__Pd

    @property
    def MUT(self) -> float:
        """
        Retrieve the mean up tick

        Returns
        -------
        float
            The mean up tick
        """
        return self.__MUT

    @property
    def MDT(self) -> float:
        """
        Retrieve the mean down tick

        Returns
        -------
        float
            The mean down tick
        """
        return self.__MDT

    @property
    def volatility(self) -> float:
        """
        Retrieve the effective Black-Scholes volatility of the index for long
        time scales

        Returns
        -------
        float
            The effective Black-Scholes volatility
        """
        return self.__volatility

    def cumul_returns_distrib(self, n: int) -> np.array:
        """
        Final cumulated returns distribution

        Parameters
        -------
        n        : int
            Number of ticks in the future  
        Returns
        -------
        pdf_n_ticks
            The final cumulated returns distribution n ticks in the future
        x
            x-values of the distribution 
        """

        def fast_exponentiation(base, exponent):
            if exponent == 0:
                result = 1
            elif exponent == 1:
                result = base
            elif exponent % 2 == 0:
                temp = fast_exponentiation(base, exponent // 2)
                result = (temp * temp)/max(temp*temp)
            else:
                temp = fast_exponentiation(base, (exponent - 1) // 2)
                result = (base * temp * temp) / max(base*temp*temp)
            return result

        def debiased(pdf, x):
            pdf = np.roll(pdf,len(x)//2 - np.argmax(pdf))
            m = sum(pdf*x*(x[1]-x[0]))
            return np.roll(pdf, -int(m/(x[1]-x[0])))

        def pdf_n_ticks_fourier(pdf,n,x):
            fourier_transform = np.fft.rfft(pdf)
            fourier_transform = fast_exponentiation(fourier_transform,n)
            inverse = np.real(np.fft.irfft(fourier_transform))
            if n%2 == 0:
                inverse= np.roll(inverse,len(x)//2)
            return inverse/np.trapz(inverse,x)

        def crash_down_pdf(x, up_prob, MUT, MDT, mu, dt):
            r = MDT*np.sqrt(dt)/mu
            x_down = [i/r for i in x]
            down_pdf = -(1-up_prob)*foldnorm.pdf(x_down,1)/r
            down_pdf[len(x)//2]+=up_prob/(x[1]-x[0])
            return down_pdf

        def crash_up_pdf(x, up_prob, MUT, MDT, mu, dt):
            r = MUT*np.sqrt(dt)/mu
            x_up = [i/r for i in x]
            up_pdf = up_prob*foldnorm.pdf(x_up,1)/r
            up_pdf[len(x)//2]+=(1-up_prob)/(x[1]-x[0])
            return up_pdf

        def crash_pdf_n_ticks(n,x,P_u,MUT,MDT,mu,dt):
            pdf_down = crash_down_pdf(x, P_u, MUT, MDT, mu, dt)
            pdf_down_n_ticks = pdf_n_ticks_fourier(pdf_down,n,x)

            pdf_up = crash_up_pdf(x, P_u, MUT, MDT, mu, dt)
            pdf_up_n_ticks = pdf_n_ticks_fourier(pdf_up,n,x)

            pdf_n_ticks = np.real(fftconvolve(pdf_down_n_ticks, pdf_up_n_ticks,mode="same"))
            pdf_n_ticks[0] = pdf_n_ticks[1]
            return pdf_n_ticks/np.trapz(pdf_n_ticks,x)

        if n>int(1e7) :
            print("n is too big, you should rather use the central limit theorem")

        l = 20*np.sqrt(n)*self.volatility*np.sqrt(dt) + 0.005
        size_pdf = 500000 # Has to be pair for the convolution
        x = np.linspace(-l, l, size_pdf)

        pdf_n_ticks = crash_pdf_n_ticks(n,x,self.up_prob,self.MUT,self.MDT,mu,dt)
        pdf_n_ticks = debiased(pdf_n_ticks, x)

        return pdf_n_ticks, x

    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
            An array of the index step returns.
        """
        rand_norm = foldnorm.rvs(1, size=size, random_state=seed)
        upOrDown = bernoulli.rvs(self.__Pu, size=size)\
            * (self.__MUT - self.__MDT) + self.__MDT
        return upOrDown * rand_norm / mu * np.sqrt(dt)

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        """
        Method that return the expected forward value of the index

        Parameters
        ----------
        S_0 : float
            The initial spot value of the index.
        t   : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index
        """
        return S_0


class BoomIndex(CrashIndex):
    """
    Boom index class
    """

    def __init__(self, index: int) -> None:
        """
        Parameters
        ----------
        index : int
            The Id number of the Crash index that represent the average
            number of second between two jumps (ex: 1000 for 'CRASH1000')
        """

        # All the data members are inherited from Crash Index and permutted
        # in order to retrieve the Boom index parameters.
        super().__init__(index)
        self._CrashIndex__Pu, self._CrashIndex__Pd,\
            self._CrashIndex__MUT, self._CrashIndex__MDT \
            = (self._CrashIndex__Pd, self._CrashIndex__Pu,
               -self._CrashIndex__MDT, -self._CrashIndex__MUT)

@geometricIndexDecorator
class JumpIndex(StochasticIndex):
    """
    Jump Index class
    """

    def __init__(
            self,
            volatility: float,
            jump_per_day: int = 72,
            jump_factor: float = 30) -> None:
        """
        Parameters
        ----------
        volatility : float
            The volatility of the index expressed in terms of percentage
            per square root of years.
        jump_per_day : int, optional
            The average number of jumps per day. Default is 72.
        jump_factor : float, optional
            The jump factor. Default is 30.
        """
        self.__volatility = volatility
        self.__jump_per_day = jump_per_day
        self.__jump_factor = jump_factor

    @property
    def volatility(self):
        """
        Retrieve the volatility of the Jump index.

        Returns
        -------
        float
            The volatility.
        """
        return self.__volatility

    @property
    def jump_per_day(self):
        """
        Retrieve the average number of jumps per day for the Jump index.

        Returns
        -------
        int
            The average number of jumps per day.
        """
        return self.__jump_per_day

    @property
    def jump_factor(self):
        """
        Retrieve the jump factor for the Jump index.

        Returns
        -------
        float
            The jump factor.
        """
        return self.__jump_factor

    @property
    def jump_probability(self):
        """
        Retrieve the probability of a jump for the Jump index.

        Returns
        -------
        float
            The probability of a jump.
        """
        return self.jump_per_day / (24 * 3600) * np.exp(-self.jump_per_day / (24 * 3600))

    @property
    def drift(self):
        """
        Retrieve the drift of the Jump index.

        Returns
        -------
        float
            The drift.
        """
        return -self.jump_factor**2 * self.volatility**2 / 2

    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Draw step return according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """
        jumpOrNot = bernoulli.rvs(self.jump_probability, size=size,
                                  random_state=seed)
        jumpSize = jumpOrNot * (self.drift * dt +
                                self.jump_factor *
                                self.volatility * np.sqrt(dt) *
                                norm.rvs(size=size))
        rand_norm = -self.volatility**2 / 2 * dt + self.volatility \
            * np.sqrt(dt) * norm.rvs(size=size, random_state=seed)
        return jumpSize + rand_norm

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        """
        Return the expected forward value of the Jump index.

        Parameters
        ----------
        S_0 : float
            The initial spot value of the index.
        t : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index.
        """
        return S_0


@geometricIndexDecorator
class DEXIndex(StochasticIndex):
    """
    DEX Index class.
    """

    def __init__(self,
                 volatility: float,
                 interest_rate: float = 0,
                 jump_frq: float = 20 * 365 * 24,
                 proba_jump_up: float = 0.8,
                 jump_up_size: float = 4e-4,
                 jump_down_size: float = 3e-3) -> None:
        """
        Parameters
        ----------
        volatility : float
            The volatility of the index expressed in terms of percentage
            per square root of years.
        interest_rate : float, optional
            The interest rate associated with the index (in terms of percentage
            per year). Default is 0.
        jump_frq : float, optional
            The average jump frequency in number of jumps per year.
            Default is 20 * 365 * 24.
        proba_jump_up : float, optional
            The probability of an upward jump. Default is 0.8.
        jump_up_size : float, optional
            The size of an upward jump. Default is 4e-4.
        jump_down_size : float, optional
            The size of a downward jump. Default is 3e-3.
        """
        self.__volatility = volatility
        self.__interest_rate = interest_rate
        self.__jump_frq = jump_frq
        self.__proba_jump_up = proba_jump_up
        self.__jump_up_size = jump_up_size
        self.__jump_down_size = jump_down_size

    @property
    def volatility(self) -> float:
        """
        Retrieve the volatility of the DEX index.

        Returns
        -------
        float
            The volatility.
        """
        return self.__volatility

    @property
    def interest_rate(self) -> float:
        """
        Retrieve the interest rate associated with the DEX index.

        Returns
        -------
        float
            The interest rate.
        """
        return self.__interest_rate

    @property
    def jump_frq(self) -> float:
        """
        Retrieve the average jump frequency in number of jumps per year for the DEX index.

        Returns
        -------
        float
            The average jump frequency.
        """
        return self.__jump_frq

    @property
    def proba_jump_up(self) -> float:
        """
        Retrieve the probability of an upward jump for the DEX index.

        Returns
        -------
        float
            The probability of an upward jump.
        """
        return self.__proba_jump_up

    @property
    def jump_up_size(self) -> float:
        """
        Retrieve the size of an upward jump for the DEX index.

        Returns
        -------
        float
            The size of an upward jump.
        """
        return self.__jump_up_size

    @property
    def jump_down_size(self) -> float:
        """
        Retrieve the size of a downward jump for the DEX index.

        Returns
        -------
        float
            The size of a downward jump.
        """
        return self.__jump_down_size

    @property
    def alpha(self) -> float:
        """
        Retrieve the alpha parameter for the DEX index.

        Returns
        -------
        float
            The alpha parameter.
        """
        return self.proba_jump_up / (1 - self.jump_up_size) \
            + (1 - self.proba_jump_up) / (1 + self.jump_down_size) - 1

    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Draw step returns according to the corresponding index logic.

        Parameters
        ----------
        dt  : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
        """

        jumpNumber = poisson.rvs(self.jump_frq * dt, size=size)
        jumpDirection = 2 * bernoulli.rvs(self.proba_jump_up, size=size) - 1
        jumpSize = np.zeros(size)
        jumpSize[(jumpNumber > 0) & (jumpDirection == 1)] = \
            np.array([np.sum(expon.rvs(scale=self.jump_up_size, size=nb))
                      for nb in jumpNumber[(jumpNumber > 0)
                                           & (jumpDirection == 1)]])
        jumpSize[(jumpNumber > 0) & (jumpDirection == -1)] = \
            -np.array([np.sum(expon.rvs(scale=self.jump_down_size, size=nb))
                       for nb in jumpNumber[(jumpNumber > 0)
                                            & (jumpDirection == -1)]])

        returns = (self.interest_rate - self.volatility**2/2
                   - self.jump_frq * self.alpha) * dt \
            + self.volatility * np.sqrt(dt) * norm.rvs(size=size) + jumpSize

        return returns

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        """
        Return the expected forward value of the DEX index.

        Parameters
        ----------
        S_0 : float
            The initial spot value of the index.
        t : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index.
        """
        return S_0


@geometricIndexDecorator
class DriftSwitchIndex(StochasticIndex):
    """
    Drift Swicth index class.

    This is the implementation of a three regimes model, with positive, null, and negative drift
    """

    def __init__(self, vol: float, drift: float, T: float) -> None:
        """
        Initialisation of Drift Switch Index

        :param vol: float
            Index volatility
        :param drift : float
            Index drift
        :param T: float
            Characteristic time of regimes
        """
        self.__vol = vol
        self.__drift = drift
        self.__T = T

    @property
    def T(self) -> float:
        return self.__T

    @property
    def drift(self) -> float:
        return self.__drift

    @property
    def vol(self) -> float:
        return self.__vol

    @property
    def gamma(self, dt: float = dt):
        """
        Gamma coefficient formula

        Parameters
        ----------
        dt : float
            Time lapse between two steps in the stochastic process.

        Returns 
        -------
        gamma : float
            The value of gamma
        """
        l = 1 / self.T
        drift = self.drift
        x_p = (l * np.exp(drift*dt)) / (1 - np.exp(drift*dt)*(1-l)) - 1
        x_m = (l * np.exp(-drift*dt)) / (1 - np.exp(-drift*dt)*(1-l)) - 1
        a_p0 = (- 2 * x_m - x_p * (1+x_m)) / (x_p - x_m)
        return a_p0

    @property
    def prob_matrix(self) -> np.array:
        """
        Transition matrix initialisation (Markov)

        Parameters
        ----------
        T : float
            Characteristic time of regimes

        Returns 
        -------
        P : np.array()
            Probability matrix 
        """
        g = self.gamma
        P = np.zeros((3, 3))
        l = 1 / self.T
        P[:, 0] = np.array([1 - l, l/2, l/2])
        P[:, 1] = np.array([(1 - g) * l, 1 - l, g * l])
        P[:, 2] = np.array([l/2, l/2, 1 - l])
        return P

    def one_d_regime_generation(self, n: int) -> np.array:
        """
        Generate a 1D array of n values of regimes between -1, 0 and 1

        Parameters
        ----------
        n : int
            The number of timesteps

        Returns 
        -------
        mu : np.array()
            The array of regimes
        """
        # Generate the durations of the regimes
        regimes_durations, k = [], 0
        while k < n :
            t = np.random.geometric(1/self.T)
            k += t
            regimes_durations.append(t)
        regimes_durations[-1] -= k-n
        regimes_begin = np.concatenate([np.zeros(1),np.cumsum(regimes_durations)[:-1]])
        s = len(regimes_durations)

        # Generate the alternance of the regimes (of duration 1)
        regimes_list, cur_regime = [], 1
        transitions = np.array([np.empty(s),np.empty(s),np.empty(s)],dtype=int)
        for i in [0,1,2]:
            trans_prob = self.prob_matrix[i]
            trans_prob[i] = 0
            trans_prob /= sum(trans_prob)
            transitions[i] = np.random.choice([0,1,2],p=trans_prob,size=s)
        for i in range(s):
            regimes_list.append(cur_regime-1)
            cur_regime = transitions[cur_regime][i]

        # Generate the final array of regimes with the good durations
        mu = np.empty(n)
        for i,regime in enumerate(regimes_list):
            t = int(regimes_begin[i])
            mu[t:t+regimes_durations[i]] = np.full(regimes_durations[i],regime)
        return mu

    def regime_generation(self, size) -> np.ndarray:
        """
        Generate an array / several arrays of regimes between -1, 0 and 1 

        Parameters
        ----------
        size : int or pair
            The number of timesteps

        Returns 
        -------
        result : np.array() or np.ndarray()
            The array of regimes
        """
        if type(size) == int :
            return self.one_d_regime_generation(size)
        (k,n) = size
        result = np.empty(size)
        for i in range(k):
            result[i] = self.one_d_regime_generation(n)
        return result

    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Generate step returns based on the stochastic process parameters

        Parameters
        ----------
        dt : float, optional
            Time lapse between two steps in the stochastic process.
            Default is dt.
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.
        seed : None or int
            The seed of the random generator. Default is None

        Returns
        -------
        np.ndarray
            Array of step returns based on the specified parameters.
        """

        x = norm.rvs(loc=0, scale=1, size=size, random_state=seed)
        regime = self.regime_generation(size)
        log_returns = (regime * self.drift - 0.5 * self.vol ** 2) * dt \
            + self.vol * x * np.sqrt(dt)
        return log_returns

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        """
        Return the expected forward value of the DEX index.

        Parameters
        ----------
        S_0 : float
            The initial spot value of the index.
        t : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index.
        """
        return S_0


class StepIndex(StochasticIndex):
    """
    Step Index class.
    """

    def __init__(self, step: float = 0.1):
        """
        Parameters
        ----------
        step : float, optional
            The step size of the index. Default is 0.1.
        """
        self.__step = step

    @property
    def step(self):
        """
        Retrieve the step size of the Step index.

        Returns
        -------
        float
            The step size.
        """
        return self.__step

    def _drawMarginalReturn(self,
                            size: Union[int, tuple[int]] = 1,
                            dt: float = dt,
                            seed: int = None) -> np.ndarray:
        """
        Generate instantaneous return random variates for the Step index.

        Parameters
        ----------
        size : int or tuple of int, optional
            Size of the array with rows and columns corresponding respectively
            to the different paths and steps of the stochastic process.
            Default is 1.

        Returns
        -------
        np.ndarray
            An array of instantaneous return random variates for the Step index.
        """
        return (2 * bernoulli.rvs(0.5, size=size) - 1) * self.step

    def expectedFwdValue(self, S_0: float, T: float) -> float:
        """
        Return the expected forward value of the Step index.

        Parameters
        ----------
        S_0 : float
            The initial spot value of the index.
        t : float
            The forward time.

        Returns
        -------
        float
            The forward value at time t of the index.
        """
        return S_0
