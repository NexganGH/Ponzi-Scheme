import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline

class DifferentialEquations:

# av_k = 5, N = 1000
    def __init__(self, N, M, lambda_, avg_k, mu, rr, rp):
        self.d, self.i, self.p = None, None, None
        self.sol_S = None
        self.N = N
        self.M = M
        self.lambda_ = lambda_
        self.avg_k = avg_k
        self.mu = mu
        self.rr = rr
        self.rp = rp
        self.t_span, self.t_eval = None, None

    def system(self, t, y):
        i, p, d = y

        lambda__ = self.lambda_(t)
        mu_ = self.mu(t)
        di_dt = lambda__ * p * self.avg_k * i - mu_ * i
        dp_dt = -lambda__ * p * self.avg_k * i
        dd_dt = mu_ * i

        return [di_dt, dp_dt, dd_dt]


    def _S_der(self, t, S):
        return S * self.rr(t) + 100 * self.N * self.lambda_(t) * self.p(t) * self.avg_k * self.i(t) - self._W(t)

    def solve_densities(self, t_start=0, t_end=30, intervals=1000):
        self.t_span = (t_start, t_end)
        self.t_eval = np.linspace(t_start, t_end, intervals)

        sol = solve_ivp(self.system, self.t_span, [0.001, 1, 0], t_eval=self.t_eval)
        t = sol.t
        i_val = sol.y[0]
        p_val = sol.y[1]
        d_val = sol.y[2]

        self.i = CubicSpline(t, i_val)
        self.p = CubicSpline(t, p_val)
        self.d = CubicSpline(t, d_val)

    def solve_money(self):
        if self.i is None:
            raise Exception('Calcola prima solve_densities')

        self.sol_S = solve_ivp(self._S_der, self.t_span, [0], t_eval=self.t_eval)
        #S_values = self.sol_S.y[0]

    def _g(self, tau, t):
        return 100 * np.exp(quad(self.rp, tau, t)[0])

    def _joined_at_time(self, tau):
        return self.lambda_(tau) * self.avg_k * self.p(tau) * self.i(tau)


    def _W(self, t):
        if t == 0:
            return 0
        else:
            return (self.N * self.mu(t) * self.i(t) * quad(lambda tau: self._joined_at_time(tau) * self._g(tau, t), 0, t)[0]
                    / quad(lambda tau: self._joined_at_time(tau), 0, t)[0])


    def graph(self, name):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        t = self.sol_S.t
        # # Plot primary variables on the left y-axis
        ax1.plot(t, self.i(t), label='Investitori (i)', color='blue')
        ax1.plot(t, self.p(t), label='Potenziali Investitori (p)', color='green')
        ax1.plot(t, self.d(t), label='Deinvestitori (d)', color='gray')
        # ax1.plot(t, d(t), label='Deinvestitori (d)', color='red')

        ax1.set_xlabel('Tempo (anni)')
        ax1.set_ylabel('Popolazione')
        ax1.legend(loc='upper left')
        ax1.grid()

        # Create secondary y-axis
        ax2 = ax1.twinx()
        #ax2.plot(t, [self._W(ti) for ti in t], label='Withdrawal', color='purple', linestyle='dashed')
        #ax2.plot(t, [av_W(ti) for ti in t], label='Average Withdrawal Value', color='red', linestyle='dashed')
        ax2.plot(t, self.sol_S.y[0], label='Money', color='red', linestyle='dashed')
        ax2.set_ylabel('Money')
        ax2.legend(loc='upper right')

        # Title
        plt.title('Evoluzione del Sistema nel Tempo')

        plt.savefig(f'imgs/{name}.png')
        plt.show()

