from simulation import InterestCalculator


class PonziParameters:
    def __init__(self,
                 M: float=1.,
                 starting_capital: float =50.,
                 lambda_ = lambda t: 0.1,
                 mu = lambda t: 0.1,
                 rp = lambda t: 0.1,
                 rr = lambda t: 0.05
                 ):
        self.M = M
        self.starting_capital = starting_capital
        self.lambda_ = lambda_
        self.mu = mu
        self.rp = rp
        self.rr = rr
        self.interest_calculator = InterestCalculator(rp=rp, rr=rr)

