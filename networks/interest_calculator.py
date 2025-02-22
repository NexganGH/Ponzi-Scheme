from scipy.integrate import quad, solve_ivp
import numpy as np

class InterestCalculator:
    def __init__(self, r_p, r_r): # ritorno promesso, ritorno realizzato, sono due funzioni del tempo.
        self.r_p = r_p
        self.r_r = r_r

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

    def func(self, t, g):
        #print(g)
        return [self.r_r(t) * g[0]]
    def realized_return(self, starting, t1, t2):
        return starting * np.exp(quad(self.r_r, t1, t2, limit=30)[0])
