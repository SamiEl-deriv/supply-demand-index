import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

class DCDPricer:
    def __init__(self):
        # Dummy volatility surface data
        self.vol_surface = {
            'AUDUSD': {7: 0.08, 14: 0.085, 30: 0.09, 90: 0.095},
            'GBPUSD': {7: 0.07, 14: 0.075, 30: 0.08, 90: 0.085},
            'EURUSD': {7: 0.065, 14: 0.07, 30: 0.075, 90: 0.08},
            'USDJPY': {7: 0.09, 14: 0.095, 30: 0.10, 90: 0.105},
            'USDCAD': {7: 0.075, 14: 0.08, 30: 0.085, 90: 0.09}
        }
        
        # Base interest rates for each currency
        self.interest_rates = {
            'USD': 0.05,  # 5% USD rate
            'AUD': 0.04,  # 4% AUD rate
            'GBP': 0.045, # 4.5% GBP rate
            'EUR': 0.035, # 3.5% EUR rate
            'JPY': 0.001, # 0.1% JPY rate
            'CAD': 0.047  # 4.7% CAD rate
        }

    def get_volatility(self, currency_pair, tenor_days):
        return self.vol_surface[currency_pair][tenor_days]

    def get_rates(self, currency_pair):
        base_ccy = currency_pair[:3]
        quote_ccy = currency_pair[3:]
        return self.interest_rates[base_ccy], self.interest_rates[quote_ccy]

    def calculate_dcd(self, currency_pair, spot_price, notional, tenor_days, strike_direction="PUT"):
        """
        Calculate DCD pricing using Black-Scholes model
        
        Parameters:
        - currency_pair: str, e.g., 'EURUSD'
        - spot_price: float, current exchange rate
        - notional: float, amount in base currency
        - tenor_days: int, duration in days
        - strike_direction: str, "PUT" or "CALL"
        
        Returns:
        - dict containing strike price, yield, and other DCD parameters
        """
        # Get parameters
        volatility = self.get_volatility(currency_pair, tenor_days)
        r_d, r_f = self.get_rates(currency_pair)  # domestic and foreign rates
        
        # Convert tenor to years
        T = tenor_days / 365.0
        
        # Calculate d1 and d2 for Black-Scholes
        sigma = volatility
        forward = spot_price * np.exp((r_d - r_f) * T)
        
        # Premium target (typically 2-3% of notional)
        premium_target = notional * 0.025  # 2.5% of notional
        
        # Binary search to find strike price that gives desired premium
        K_low = spot_price * 0.5
        K_high = spot_price * 1.5
        
        for _ in range(50):  # Maximum iterations for binary search
            K = (K_low + K_high) / 2
            
            d1 = (np.log(spot_price/K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if strike_direction == "PUT":
                option_price = (K * np.exp(-r_d * T) * norm.cdf(-d2) - 
                              spot_price * np.exp(-r_f * T) * norm.cdf(-d1))
            else:  # CALL
                option_price = (spot_price * np.exp(-r_f * T) * norm.cdf(d1) - 
                              K * np.exp(-r_d * T) * norm.cdf(d2))
            
            if abs(option_price - premium_target) < 0.0001:
                break
            elif option_price > premium_target:
                if strike_direction == "PUT":
                    K_low = K
                else:
                    K_high = K
            else:
                if strike_direction == "PUT":
                    K_high = K
                else:
                    K_low = K
        
        # Calculate enhanced yield
        base_yield = r_d if strike_direction == "PUT" else r_f
        option_premium_yield = (premium_target / notional) * (365 / tenor_days)
        enhanced_yield = base_yield + option_premium_yield
        
        return {
            'strike_price': round(K, 4),
            'enhanced_yield': round(enhanced_yield * 100, 2),  # Convert to percentage
            'volatility': round(volatility * 100, 2),  # Convert to percentage
            'base_yield': round(base_yield * 100, 2),  # Convert to percentage
            'premium_yield': round(option_premium_yield * 100, 2),  # Convert to percentage
            'option_premium': round(premium_target, 2),
            'spot_reference': round(spot_price, 4)
        }
