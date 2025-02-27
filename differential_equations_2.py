import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline

class DifferentialEquations2:

# av_k = 5, N = 1000
    def __init__(self, N, M, lambda_, avg_k, mu, rr, rp, ponzi_capital):
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
        self.ponzi_capital=ponzi_capital

    def system(self, t, y):
        i, p, d, S, U, C = y

        lambda__ = self.lambda_(t)
        mu_ = self.mu(t)
        di_dt = lambda__ * p * self.avg_k * i - mu_ * i
        dp_dt = -lambda__ * p * self.avg_k * i
        dd_dt = mu_ * i

        avg_w = U / C if C != 0 else 0
        dS_dt = ((S * self.rr(t)
                 + self.M * self.N * self.lambda_(t) * p * self.avg_k * i)
                 - self.N * self.mu(t) * i * avg_w)
        dU_dt = lambda__ * self.avg_k * p * i * self.M + (self.rp(t) - mu_) * U
        dC_dt = lambda__ * self.avg_k * p * i - mu_ * C


        return [di_dt, dp_dt, dd_dt, dS_dt, dU_dt, dC_dt]



    def solve_densities(self, t_start=0, t_end=30, intervals=1000):
        self.t_span = (t_start, t_end)
        self.t_eval = np.linspace(t_start, t_end, intervals)

        sol = solve_ivp(self.system, self.t_span, [1/self.N, 1, 0, self.ponzi_capital, 0, 0], t_eval=self.t_eval, rtol=1e-8,  atol=1e-10, max_step=0.01)
        self.t = sol.t
        i_val = sol.y[0]
        p_val = sol.y[1]
        d_val = sol.y[2]
        self.sol_S = sol.y[3]

        self.i = CubicSpline(self.t, i_val)
        self.p = CubicSpline(self.t, p_val)
        self.d = CubicSpline(self.t, d_val)


    def graph(self, name):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        t = self.t
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
        ax2.plot(t, self.sol_S, label='Money', color='red', linestyle='dashed')
        ax2.set_ylabel('Money')
        ax2.legend(loc='upper right')

        # Title
        plt.title('Evoluzione del Sistema nel Tempo')

        plt.savefig(f'imgs/{name}.png')
        plt.show()

