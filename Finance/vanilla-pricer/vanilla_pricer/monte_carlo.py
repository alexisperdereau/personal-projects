import numpy as np
import math
# Monte Carlo simulation for option pricing
# Monte Carlo simulation is a numerical method used to estimate the price of options by simulating random paths of the underlying asset price
# It is particularly useful for pricing complex derivatives and options with path-dependent features
# It uses random sampling to obtain numerical results and is based on the law of large numbers

class MonteCarlo:
    def __init__(self, option, num_simulations):
        self.option = option
        self.num_simulations = num_simulations

    def price(self):
        """
        Monte Carlo simulation involves simulating a large number of paths of the underlying asset price and calculating the average payoff:

        Formula:
        - S_T = S * exp[(r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z]
        Where Z ~ N(0,1) is a standard normal random variable.
        
        Payoff for a call option: max(S_T - K, 0)

        The option price is then:
        Option Price = exp(-r * T) * (1 / N) * sum(payoffs)

        """
        np.random.seed(0)  # Set seed for reproducibility
        payoff_sum = 0
        for _ in range(self.num_simulations):
            z = np.random.normal(0, 1)  # Standard normal random variable
            ST = self.option.S * np.exp((self.option.r - 0.5 * self.option.sigma**2) * self.option.T + 
                                        self.option.sigma * math.sqrt(self.option.T) * z)
            payoff_sum += max(ST - self.option.K, 0)
        
        return math.exp(-self.option.r * self.option.T) * (payoff_sum / self.num_simulations)
