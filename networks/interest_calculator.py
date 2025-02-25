from scipy.integrate import quad, solve_ivp
import scipy.interpolate as interp
import numpy as np

class InterestCalculator:
    def __init__(self, r_p, r_r): # ritorno promesso, ritorno realizzato, sono due funzioni del tempo.
        self.r_p = r_p
        self.r_r = r_r
        self.market_positivity = None

    def func2(self, t, g):
        return [self.r_p(t) * g[0]]

    def promised_return_at_time(self, starting, t1, t2):
        """
        Calcola il ritorno promesso al tempo t2 con un capitale iniziale di starting.
        :param starting:
        :param t1:
        :param t2:
        :return:
        """
        return starting * np.exp(quad(self.r_p, t1, t2, limit=30)[0])

    def realized_return(self, starting, t1, t2):
        return starting * np.exp(quad(self.r_r, t1, t2, limit=30)[0])

    def compute_market_positivity(self, t1, t2, num_points=100, alpha=np.log(2) / 2):
        """
        Compute market positivity over a discretized interval and interpolate it.

        :param r_r: Function r_r(t) representing returns.
        :param t1: Start time.
        :param t2: End time.
        :param num_points: Number of discretized intervals.
        :param alpha: Decay rate for weighting.
        :return: Interpolated function for market positivity.
        """
        t_values = np.linspace(t1, t2, num_points)
        positivity_values = np.array([self._market_positivity_cont(t=t, rr=self.r_r, alpha=alpha) for t in t_values])

        # Use cubic spline interpolation
        spline = interp.CubicSpline(t_values, positivity_values)

        self.market_positivity =  spline

    def _market_positivity_cont(self, t, rr, alpha=np.log(2) / 2):
        """
        Compute market positivity based on past returns with an exponential weighting.

        :param t: Current time.
        :param alpha: Decay factor for weighting.
        :return: Market positivity score.
        """
        t_start = t - 1  # consider up to one year in the past
        t_end = t
        integrand = lambda ti: np.exp(-alpha * (t_end - ti)) * rr(ti)

        # Compute the integral using quad
        positivity= quad(integrand, t_start, t_end)[0]
        # normalizzando
        positivity /= quad(lambda ti: np.exp(-alpha * (t_end - ti)), t_start, t_end)[0]

        return positivity


    def mu_from_market_positivity(self, market_pos, base=0.1, min=0.05, max=0.3, steepness=5):
        return (max - min) / (1 + np.exp(-steepness * (-market_pos)))

    def lambda_from_market_positivity(self, market_pos, base=0.1, min=0.05, max=0.3, steepness=5):
        return (max - min) / (1 + np.exp(-steepness * market_pos))


    def mu_from_rr_func(self, base=0.1, min=0.05, max=0.2, steepness=3):
        """
        Calcola il la funzione mu al variare del tempo considerando il ritorno reale. Ad esempio, si può pensare che
        se il mercato finanziario vada male allora più investitori usciranno dallo schema perché hanno paura o perché
        devono pagare altri debiti.
        """

        #return lambda t: base - min - (max-min)/(1 + np.exp(-steepness * self.market_positivity(t)))
        return lambda t: self.mu_from_market_positivity(self.market_positivity(t), base=base, min=min, max=max, steepness=steepness)

        #return lambda t: (max - min) / (1 + np.exp(-steepness * self.market_positivity(t)))

    def lambda_from_rr_func(self, base=0.1, min=0.05, max=0.2, steepness=3):
        return lambda t: self.lambda_from_market_positivity(self.market_positivity(t), base=base, min=min, max=max,
                                                        steepness=steepness)
