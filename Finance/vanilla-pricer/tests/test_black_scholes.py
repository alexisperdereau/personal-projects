import pytest
from vanilla_pricer.option import Option
from vanilla_pricer.black_scholes import BlackScholes

def test_black_scholes():
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2)
    model = BlackScholes(option)
    # The price calculated manually should be approximately 10.4506
    assert abs(model.price() - 10.4506) < 0.01
