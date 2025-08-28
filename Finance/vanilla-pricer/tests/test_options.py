from vanilla_pricer.option import Option

def test_option() :
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert option.S == 100
    assert option.K == 100
    assert option.T == 1
    assert option.r == 0.05
    assert option.sigma == 0.2