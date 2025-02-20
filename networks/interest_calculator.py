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
        return starting * np.exp(quad(self.r_p, t1, t2)[0])
       # sol = solve_ivp(self.func2, (t1, t2), [starting], t_eval=[t2])
        # val = starting * np.e ** quad(lambda t: self.r_r(t), t1, t2)[0]
        #print(f'realized returned: from t_1 {t1} to t_2 {t2}: {sol.y[0][-1]}')
     #   return 0
        #return sol.y[0][-1]

    def func(self, t, g):
        #print(g)
        return [self.r_r(t) * g[0]]
    def realized_return(self, starting, t1, t2):
        #print(f'calc return from {t1} to {t2}, with interest: {self.r_r(t1)}, res:', starting, ' to ', starting * np.exp(quad(self.r_r, t1, t2)[0]))
        return starting * np.exp( quad(self.r_r, t1, t2)[0])
        #return starting * np.exp(quad(self.r_r, t1, t2, limit=30)[0])
        #return starting*np.e**(self.r_r(t2)*(t2 - t1))#starting + starting*self.r_r(t2)*(t2-t1)