from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import numpy as np

# i, p, d, g, F

def lambda_(t):
    return 0.25 #10% annuo

def mu(t):
    return 0.25 #5% annuo

def rp(t):
    return 0.1 #10% annnuo

intervals = 1000
# Supponiamo che t sia espresso in mesi

av_k = 5

ol

def inte2(t_):
    return mu(t_)

def inte1(tau, p):

    return lambda_(tau) * p * np.exp(-quad(inte2, tau, t)[0]) * g(t)
def pde_system(t, y):
    i, p, d, g, f = y

    lambda__ = lambda_(t)
    mu_ = mu(t)
    di_dt = lambda__ * p * av_k * i - mu_ * i
    dp_dt = -lambda__ * p * av_k * i
    dd_dt = mu_ * i
    dg_dt = rp(t)  * g
    df_dt = di_dt - mu(t) * quad(lambda tau: inte1(tau, p), 0, t)[0]

    return [di_dt, dp_dt, dd_dt, dg_dt, df_dt]

t_span = (0., 30.)

W = 1.
sol = solve_ivp(pde_system, t_span, [0.001, 1, 0, W], t_eval=np.linspace(0, 20, intervals))
t = sol.t
i = sol.y[0]
p = sol.y[1]
d = sol.y[2]
g = sol.y[3]
plt.figure(figsize=(10, 6))
plt.plot(t, i, label='Investitori (i)')
plt.plot(t, p, label='Potenziali Investitori (p)')
plt.plot(t, d, label='Deinvestitori (d)')
plt.plot(t, g, label='Capitale (g)')
plt.xlabel('Tempo (anni)')
plt.ylabel('Valore')
plt.title('Evoluzione del Sistema nel Tempo')
plt.legend()
plt.grid()
plt.show()