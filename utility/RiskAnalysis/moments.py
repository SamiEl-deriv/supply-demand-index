import numpy as np
import pandas as pd
from scipy import stats


class MomentAnalyzer:
    """Class for calculating statistical moments of return distributions."""
    
    def __init__(self):
        self._moments = None

    @property
    def moments(self):
        if self._moments is None:
            raise AttributeError("Moments have not been calculated. Use calculate_moments first.")
        return self._moments

    @moments.setter
    def moments(self, value):
        self._moments = value

    def calculate_static_moments(self, returns, annualize=True):
        """
        Calculate static moments for the entire return series.

        Args:
            returns (pd.Series): Series of returns.
            annualize (bool): Whether to annualize the moments.

        Returns:
            dict: Dictionary containing moment calculations.
        """
        log_returns = np.log(1 + returns)
        ann_factor = 252 if annualize else 1
        n = len(log_returns)
        
        # Calculate basic moments
        mean = log_returns.mean()
        variance = log_returns.var()
        skewness = ((log_returns - mean) ** 3).mean() / variance ** 1.5
        kurtosis = ((log_returns - mean) ** 4).mean() / variance ** 2
        excess_kurtosis = kurtosis - 3
        
        # Jarque-Bera test
        jb_stat = n * (skewness**2 / 6 + excess_kurtosis**2 / 24)
        jb_pvalue = 1 - stats.chi2.cdf(jb_stat, df=2)
        
        # Annualize if requested
        if annualize:
            mean *= ann_factor
            variance *= ann_factor
            skewness /= np.sqrt(ann_factor)
            kurtosis /= ann_factor
            excess_kurtosis /= ann_factor
        
        self.moments = {
            'mean': mean,
            'variance': variance,
            'volatility': np.sqrt(variance),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal': jb_pvalue > 0.05
        }
        return self.moments

    def calculate_rolling_moments(self, returns, window, annualize=True, check_convergence=True):
        """
        Calculate rolling moments over a specified window.

        Args:
            returns (pd.Series): Series of returns.
            window (int): Rolling window size.
            annualize (bool): Whether to annualize the moments.

        Returns:
            dict: Dictionary containing rolling moment series.
        """
        log_returns = np.log(1 + returns)
        ann_factor = 252 if annualize else 1
        
        # Calculate rolling moments
        rolling_mean = log_returns.rolling(window=window).mean() * ann_factor
        rolling_var = log_returns.rolling(window=window).var() * ann_factor
        rolling_vol = np.sqrt(rolling_var)
        
        def rolling_moment(series, window, moment):
            return series.rolling(window=window).apply(
                lambda x: ((x - x.mean()) ** moment).mean() / (x.var() ** (moment/2))
            )
        
        rolling_skew = rolling_moment(log_returns, window, 3) / np.sqrt(ann_factor)
        rolling_kurt = rolling_moment(log_returns, window, 4) / ann_factor
        rolling_excess_kurt = rolling_kurt - 3
        
        moments_dict = {
            'mean': rolling_mean,
            'variance': rolling_var,
            'volatility': rolling_vol,
            'skewness': rolling_skew,
            'kurtosis': rolling_kurt,
            'excess_kurtosis': rolling_excess_kurt
        }
        
        if check_convergence:
            convergence_metrics = self.analyze_convergence(moments_dict)
            moments_dict.update(convergence_metrics)
        
        self.moments = moments_dict
        return self.moments

    def analyze_convergence(self, moments_dict, threshold=0.05, stability_window=20):
        """
        Analyze the convergence of moment estimates.
        
        Args:
            moments_dict (dict): Dictionary of rolling moment series.
            threshold (float): Threshold for considering a moment converged.
            
        Returns:
            dict: Convergence metrics for each moment.
        """
        convergence_dict = {}
        
        # For each moment, analyze its convergence
        for name, series in moments_dict.items():
            # Skip if not enough data
            if len(series) < 10:
                continue
                
            # Calculate rate of change (percentage change over last stability_window periods)
            roc = abs(series.pct_change(stability_window).iloc[-1])
            
            # Calculate stability (standard deviation of last stability_window values relative to mean)
            recent_values = series.iloc[-stability_window:]
            recent_stability = recent_values.std() / abs(recent_values.mean()) if abs(recent_values.mean()) > 1e-10 else 0
            
            # Determine if converged based on both rate of change and stability
            is_converged = (roc < threshold) and (recent_stability < threshold)
            
            convergence_dict[f'{name}_convergence'] = {
                'is_converged': is_converged,
                'rate_of_change': roc,
                'stability': recent_stability
            }
        
        # Add overall convergence status
        all_converged = all(metrics['is_converged'] 
                          for metrics in convergence_dict.values())
        
        convergence_dict['overall_convergence'] = {
            'is_converged': all_converged,
            'convergence_threshold': threshold,
            'non_converged_moments': [
                name.replace('_convergence', '') 
                for name, metrics in convergence_dict.items()
                if not metrics['is_converged'] and name != 'overall_convergence'
            ]
        }
        
        return {'convergence': convergence_dict}
