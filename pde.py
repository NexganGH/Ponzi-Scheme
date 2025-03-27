import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline


# i, p, d, g, F

def lambda_(t):
    return 0.25 #10% annuo

def mu(t):
    return 0.50 #5% annuo


def rr(t):
    return 0.1

def rp(t):
    return 0.1
    #return 0.1 #10% annnuo

intervals = 1000
# Supponiamo che t sia espresso in mesi

av_k = 5

def pde_system(t, y):
    i, p, d = y

    lambda__ = lambda_(t)
    mu_ = mu(t)
    di_dt = lambda__ * p * av_k * i - mu_ * i
    dp_dt = -lambda__ * p * av_k * i
    dd_dt = mu_ * i

    return [di_dt, dp_dt, dd_dt]

t_span = (0., 30.)

W = 1.
sol = solve_ivp(pde_system, t_span, [0.001, 1, 0], t_eval=np.linspace(0, 30, intervals))
t = sol.t
i_val = sol.y[0]
p_val = sol.y[1]
d_val = sol.y[2]

i = CubicSpline(t, i_val)
p = CubicSpline(t, p_val)
d = CubicSpline(t, d_val)

def g(tau, t):
    return 100*np.exp(quad(rp, tau, t)[0])


N = 1000

def joined_at_time_and_hasnt_left(tau, t):
    return lambda_(tau) * av_k * p(tau) * i(tau) * np.exp(-1 * quad(lambda ti: mu(ti), tau, t)[0])

def W(t):
    if t == 0: return 0
    else:
        return (N * mu(t) * i(t) * quad(lambda tau: joined_at_time_and_hasnt_left(tau, t) * g(tau, t) , 0, t)[0]
            / quad(lambda tau: joined_at_time_and_hasnt_left(tau, t), 0, t)[0])

def av_W(t):
    return W(t) / (N * mu(t) * i(t))

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot primary variables on the left y-axis
ax1.plot(t, i(t), label='Investitori (i)', color='blue')
ax1.plot(t, p(t), label='Potenziali Investitori (p)', color='green')
#ax1.plot(t, d(t), label='Deinvestitori (d)', color='red')

ax1.set_xlabel('Tempo (anni)')
ax1.set_ylabel('Popolazione')
ax1.legend(loc='upper left')
ax1.grid()



# money flux
def S_der(t, S):
    return S * rr(t) +  100 * N * lambda_(t) * p(t) * av_k * i(t) -  W(t)
    #return 100 * N * lambda_(t) * p(t) * av_k * i(t) -  W(t)

sol_S = solve_ivp(S_der, t_span, [0], t_eval=np.linspace(0, 30, intervals))
S_values = sol_S.y[0]

# Create secondary y-axis
ax2 = ax1.twinx()
ax2.ba_plot(t, [W(ti) for ti in t], label='Withdrawal', color='purple', linestyle='dashed')
ax2.ba_plot(t, [av_W(ti) for ti in t], label='Average Withdrawal Value', color='red', linestyle='dashed')
ax2.ba_plot(t, S_values, label='Money', color='green', linestyle='dashed')
ax2.set_ylabel('Money')
ax2.legend(loc='upper right')

#ax2.plot(t, [g(0, ti) for ti in t])

# Title
plt.title('Evoluzione del Sistema nel Tempo')

plt.show()

print(quad(W, 0, 30)[0])