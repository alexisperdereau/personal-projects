import math

class BinomialTree:
    def __init__(self, option, steps):
        self.option = option
        self.steps = steps
        self.dt = self.option.T / self.steps
        self.u = math.exp(self.option.sigma * math.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (math.exp(self.option.r * self.dt) - self.d) / (self.u - self.d)  # Probability of up

    def price(self):
        """
        The Binomial Tree method involves creating a binomial tree for option pricing:
        
        Formula:
        - Up factor (u) = exp(sigma * sqrt(dt))
        - Down factor (d) = 1 / u
        - Risk-neutral probability (p) = [exp(r * dt) - d] / (u - d)

        At each step, the option price is recalculated using:
        Option price at maturity = max(S_T - K, 0) for a call option
        Option price at each prior node = [p * price(up) + (1 - p) * price(down)] * exp(-r * dt)

        """
        # Terminal values (payoffs)
        payoff = [max(self.option.S * self.u**i * self.d**(self.steps-i) - self.option.K, 0) for i in range(self.steps+1)]
        
        # Move backwards through the tree
        for j in range(self.steps - 1, -1, -1):
            for i in range(j + 1):
                payoff[i] = (self.p * payoff[i+1] + (1 - self.p) * payoff[i]) * math.exp(-self.option.r * self.dt)
        
        return payoff[0]