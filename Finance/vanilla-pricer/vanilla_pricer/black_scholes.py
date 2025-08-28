import math
from scipy.stats import norm

class BlackScholes:
    def __init__(self, option):
        self.option = option
    
    def price(self):
        """
        The Black-Scholes formula for the price of a European call option:
        
        Formula:
        d1 = [ln(S/K) + (r + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        Call option price (C) = S * N(d1) - K * exp(-r * T) * N(d2)
        Where:
        S = spot price of the asset
        K = strike price
        r = risk-free rate
        T = time to maturity
        sigma = volatility
        N(x) = cumulative distribution function of the standard normal distribution

        """
        d1 = (math.log(self.option.S / self.option.K) + 
              (self.option.r + 0.5 * self.option.sigma ** 2) * self.option.T) / (self.option.sigma * math.sqrt(self.option.T))
        d2 = d1 - self.option.sigma * math.sqrt(self.option.T)
        
        call_price = (self.option.S * norm.cdf(d1) - self.option.K * math.exp(-self.option.r * self.option.T) * norm.cdf(d2))
        return call_price
