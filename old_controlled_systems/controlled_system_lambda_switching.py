import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# System parameters
k = 5
min_lambda = 0
max_lambda = 0.4

def compute_lambda(u):
    return np.clip(u, min_lambda, max_lambda)

def dynamic_mu(ti):
    return np.where(ti < 17.5, 0.15, np.where(ti <= 20, 0.7, 0.15))


def system_dynamics(t, x, u):
    i, p = x
    mu = dynamic_mu(t)
    lambda_ = compute_lambda(u)
    di_dt = p * lambda_ * k * i - mu * i
    dp_dt = -p * lambda_ * k * i
    return np.vstack([di_dt, dp_dt])  # Output con forma (2, N)

def costate_equations(t, lambda_pars, x, u):
    i, p = x
    lambda_1, lambda_2 = lambda_pars
    lambda_ = compute_lambda(u)
    dlambda1_dt = lambda_1 * (p * k * lambda_ - dynamic_mu(t)) - lambda_2 * p * k * lambda_
    dlambda2_dt = lambda_1 * k * lambda_ * i - lambda_2 * k * lambda_ * i
    return np.vstack([-dlambda1_dt*0, -dlambda2_dt*0])  # Output con forma (2, N)


def optimal_control(t, lambda_pars, x):
    lambda_1, lambda_2 = lambda_pars
    switching_func = lambda_1 - lambda_2
    return max_lambda
    #return np.where(switching_func > 0, max_lambda, min_lambda)


def system_with_costates(t, y):
    x = y[:2]
    lambda_pars = y[2:]
    u = optimal_control(t, lambda_pars, x)
    dx_dt = system_dynamics(t, x, u)
    d_lambda_dt = costate_equations(t, lambda_pars, x, u)
    return np.vstack([dx_dt, d_lambda_dt])  # Forma (4, N)


i0 = 0.1
p0 = 1 - i0
lambda1_0 = 0
lambda2_0 = 0
def bc(ya, yb):
    return np.array([
        ya[0] - i0,
        ya[1] - p0,
        yb[2]-1,
        yb[3]
    ])


y0 = [i0, p0, lambda1_0, lambda2_0]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 2000)

i_guess = np.linspace(0.1, 0.2, len(t_eval))
p_guess = np.linspace(p0, p0, len(t_eval))
lambda1_guess = np.linspace(1, 1, len(t_eval))
lambda2_guess = np.linspace(0, 0, len(t_eval))
y_guess = np.vstack([i_guess, p_guess, lambda1_guess, lambda2_guess])

solution_optimal = spi.solve_bvp(system_with_costates, bc, t_eval, y_guess)
print(solution_optimal.message)
t = solution_optimal.x
i_opt, p_opt, lambda1_opt, lambda2_opt = solution_optimal.y
u_opt = np.array([optimal_control(t[j], [lambda1_opt[j], lambda2_opt[j]], [i_opt[j], p_opt[j]]) for j in range(len(t))])

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, i_opt, label="i(t) (Optimal control)", color="b")
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, p_opt, label="p(t) (Optimal control)", color="b")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, u_opt, label="u(t) (Optimal control)", color="b")
plt.xlabel("Time")
plt.ylabel("u(t)")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(t, lambda1_opt, label="$\\lambda_1(t)$", color="b")
plt.plot(t, lambda2_opt, label="$\\lambda_2(t)$", color="r")
plt.xlabel("Time")
plt.ylabel("Costate Variables")
plt.legend()
plt.title("Costate Dynamics Over Time")
plt.show()
