from vanilla_pricer.option import Option
from vanilla_pricer.binomial_tree import BinomialTree

def test_binomial_tree():
    # Define an at-the-money European call option
    # Spot price (S) = 100, Strike price (K) = 100 -> "at the money"
    # Time to maturity (T) = 1 year
    # Risk-free interest rate (r) = 5%
    # Volatility (sigma) = 20%
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2)

    # Initialize the Binomial Tree model with 1000 steps for better accuracy
    model = BinomialTree(option, steps=1000)

    # Compute the price using the Binomial Tree method
    price = model.price()

    # For this configuration, the theoretical price around 10.45
    # We accept a small error margin since the Binomial method is an approximation
    assert abs(price - 10.45) < 0.5
