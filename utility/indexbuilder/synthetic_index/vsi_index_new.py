from stochastic_process_base import StochasticProcess
import numpy as np
from typing import List, Optional

# NOTE = this Code functionallyu is same as vsi_index.py code , only with minor changes pertaining to mathematical convetions namely:
# 1 - Addition of Rows of Transition Matirx == 1
# 2 - To run this cod properly, change 'P' in process_dict.py , such that same Row Probability sum is 1

class VolatilitySwitchIndex_new(StochasticProcess):
    def __init__(self, 
                 vol: List[float],
                 drift:List[float],
                 T: List[float],
                 zeta: Optional[List[float]] = None,
                 P: Optional[np.array] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.__vol = np.array(vol)    # List of volatilities for each regime.
        self.__T = np.array(T)
        assert len(self.__vol) == len(self.__T), "Number of volatility times must be equal."
        self.__dim = len(self.__vol)   # dimensions
        if zeta : self.__zeta = np.array(zeta)    #   Initial regime / state probabilities.
        else : 
            self.__zeta = np.zeros(shape = (self.__dim))
            self.__zeta[0] = 1   # if no values provided --> assume we begin from regime-1 (index 0)
            
        assert len(self.__zeta) == self.__dim
        assert np.isclose(np.sum(self.__zeta), 1), "Sum of zeta probabilities must be 1"
        
        self.__P = self._initialize_transition_matrix(P)  # initialize transition probability matrix
        self.__current_regime = self.choose_initial_regime()  # initialize first regime/state

    def _initialize_transition_matrix(self, P: Optional[np.array]) -> np.array:
        # Check if transition matrix is provided, else generate random transition matrix
        if P is None:
            lam = [1/t for t in self.__T]
            P = np.empty(shape = (self.__dim,self.__dim))
            for i in range(self.__dim):
                for j in range(self.__dim):
                    if i == j:
                        P[i,j] = 1- lam[i]
                    else:
                        P[i,j] = lam[j]/(self.__dim - 1)
            P = P.transpose()
            # P = [[1-lam[0],lam[1]/2,lam[2]/2],[lam[0]/2,1-lam[1],lam[2]/2],[lam[0]/2,lam[1]/2,1-lam[2]]]
            # P = np.array(P)
            # P = np.random.rand(self.__dim, self.__dim)  # Generate random transition matrix
            # P /= P.sum(axis=1)[:, np.newaxis]  # Normalize rows to ensure sum of each row is 1
        P = np.array(P)
        assert isinstance(P, np.ndarray), "Transition matrix must be a numpy array."
        assert P.shape == (self.__dim, self.__dim), "Transition matrix dimensions do not match the number of regimes."
        assert np.allclose(P.sum(axis=1), 1), "Rows of transition matrix must sum to 1." # axis=0 --> columns ; axis=1 --> rows
        return P

    @property
    def volatility(self):
        return self.__vol
    @property
    def zeta_probability(self):
        return self.__zeta
    @property
    def transition_prob_matrix(self):
        return self.__P
    @property
    def current_regime(self):
        return self.__current_regime

    def choose_initial_regime(self):
        return np.random.choice(range(self.__dim), p=self.__zeta)
    
    def choose_next_regime(self):
        # Get the row corresponding to the current state
        transition_probs = self.__P[self.__current_regime]
        # Sample from the probability distribution
        next_regime = np.random.choice(range(self.__dim), p=transition_probs)
        return next_regime

    def new_return(self):
        self.__current_regime = self.choose_next_regime()
        volatility = self.__vol[self.__current_regime]
        self.log_returns_mean = ( - 0.5 * volatility**2) * self.dt
        self.log_returns_variance = volatility**2 * self.dt
        log_return = np.random.normal(self.log_returns_mean, np.sqrt(self.log_returns_variance))
        return log_return


