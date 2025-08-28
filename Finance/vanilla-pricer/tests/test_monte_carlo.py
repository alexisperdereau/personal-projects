from vanilla_pricer.option import Option
from vanilla_pricer.monte_carlo import MonteCarlo

def test_monte_carlo():
    # Define the same European call option as in the previous test
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2)

    # Initialize the Monte Carlo model with 100,000 simulations
    # A large number of simulations helps reduce random error
    model = MonteCarlo(option, num_simulations=100_000)

    # Compute the price using the Monte Carlo method
    price = model.price()

    # Again, theoretical Black-Scholes price ≈ 10.45
    # Monte Carlo results are noisy, so we allow a wider margin
    assert abs(price - 10.45) < 1.0
