class Option:
    def __init__(self, S, K, T, r, sigma):
        """
        S  : Current price of the underlying asset
        K  : Strike price
        T  : Time to maturity (in years)
        r  : Risk-free interest rate
        sigma : Volatility of the underlying asset
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def price(self):
        """
        Method to be overridden for pricing using different models
        """
        raise NotImplementedError("Pricing model is not implemented.")
