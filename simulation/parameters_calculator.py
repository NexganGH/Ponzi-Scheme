from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline
import numpy as np

class ParameterCalculator:
    def __init__(self, rp, rr): # ritorno promesso, ritorno realizzato, sono due funzioni del tempo.
        self.rp = rp
        self.rr = rr
        self._sentiment = None

    def promised_return_at_time(self, starting, t1, t2):
        """
        Calcola il ritorno promesso al tempo t2 con un capitale iniziale di starting.
        :param starting:
        :param t1:
        :param t2:
        :return:
        """
        return starting * np.exp(quad(self.rp, t1, t2, limit=30)[0])

    def ponzi_earnings(self, starting, t1, t2):
        return starting * np.exp(quad(self.rr, t1, t2, limit=30)[0])

    def compute_sentiment(self, t1, t2, num_points=100, alpha=np.log(2) / 2):
        """
        Calcola il sentimento.

        :param r_r: r_r(t): ritorni reali.
        :param t1: Tempo iniziale.
        :param t2: Tempo finale.
        :param num_points: Intervalli discreti.
        :param alpha: Decay rate for weighting.
        :return: Interpolated function for market positivity.
        """
        alpha, beta, gamma = 0.5, 1, 0.3
        def sentiment(t, m):
            val = 0 if self.rr(t) > 0 else (-self.rr(t))
            return -alpha * m + beta * (self.rr(t) ) - gamma * np.heaviside(-self.rr(t), 1) * val#-alpha * m * 1 / (1 + np.exp(-20*m))  + beta * (self.rr(t) - 0.04) - gamma * np.heaviside(-self.rr(t), 1) * val
        t_values = np.linspace(t1, t2, num_points)


        sol = solve_ivp(sentiment, [t1, t2], [0], t_eval=t_values, method='RK45')

        sentiment_values = sol.y[0]
        m_max = np.max(np.abs(sentiment_values))  # Largest absolute value
        sentiment_values = sentiment_values / (m_max + 1e-6) # normalizza

        self._sentiment = CubicSpline(sol.t, sentiment_values)
        return self._sentiment
    def get_sentiment(self):
        if self._sentiment is None:
            raise 'Calcola il sentiment con compute_sentiment'
        return self._sentiment

    def mu_from_rr(self, min=0.05, base=0.1, max=0.8, steepness=5):
        sent = self.get_sentiment()
        return lambda t: self._mu_from_sentiment(sent(t), min=min, base=base, max=max, steepness=steepness)

    def _mu_from_sentiment(self, sent, min=0.05, base=0.1, max=0.8, steepness=5):
        c = -1/steepness * np.log((max-base)/(base-min))
        return min + (max-min)/(1+np.exp(steepness*(sent - c)))
    def lambda_from_rp(self, rp, min_rp=0., base_rp=0.08, max_rp=0.25, min_lambda=0., max_lambda=0.5):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        return min_lambda + (max_lambda - min_lambda) * sigmoid(50 * (rp-base_rp))


    def mu_prom_from_rp(self, rp, base_rp=0.1, max_rp=0.25, min_mu=0., max_mu=0.5):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        return min_mu + (max_mu - min_mu) * sigmoid(-50 * (rp-base_rp) )