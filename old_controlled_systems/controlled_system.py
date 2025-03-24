import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from simulation.finance_data import FinanceData
from simulation.parameters_calculator import ParameterCalculator

data = FinanceData()
data.download()
interest_calculator = ParameterCalculator(rp=lambda t:0.1, rr=data.market_rr)
interest_calculator.compute_sentiment(0, 30, 100)
#dynamic_mu = #interest_calculator.mu_from_rr_func(min=0.05, base=0.1, max=0.5, steepness=100)
def dynamic_mu(ti):
    if ti < 17.5:
        return 0.15
    elif 17.5 <= ti <= 20:
        return 0.4
    else:
        return 0.15
# System parameters
lambda_ = lambda _: 0.2#interest_calculator.lambda_from_rr_func(max=0.3, steepness=10)
k = 5

max_up=0.85
max_down=0.1

# Define system dynamics
def system_dynamics(t, x, u):
    i, p = x  # State variables
    mu = dynamic_mu(t) + u  # Withdrawal rate

    di_dt = p * lambda_(t) * k * i - mu * i
    dp_dt = -p * lambda_(t) * k * i

    return [di_dt, dp_dt]

i_0 = 0.3
def dL_di(i):
    if np.abs(i - i_0) < 0.03:
        return 500
    else:
        return 0

# Define costate equations
def costate_equations(t, lambda_pars, x, u):
    i, p = x  # State variables
    lambda_1, lambda_2 = lambda_pars  # Costate variables

    # Compute derivatives of costates
    dlambda1_dt = lambda_1 * (p * k * lambda_(t) - (dynamic_mu(t) + u)) - lambda_2 * p * k * lambda_(t) - dL_di(i)#-lambda_1 * (lambda_p * lambda_ * k - (0.25 + u))
    dlambda2_dt = lambda_1 * k * lambda_(t) * i - lambda_2 * k * lambda_(t) * i

    return [dlambda1_dt, dlambda2_dt]


# Define optimal control function with constraints
#def optimal_control(lambda_1, i):
 #   u_unconstrained = -lambda_1 * i  # Derived from dH/du = 0
  #  u_min, u_max = -0.2, 0.2  # Control bounds
   # return np.clip(u_unconstrained, u_min, u_max)  # Apply bounds
def lagrangian(t, x):
    i, p = x
    if i > i_0:
        return 0
    return 1

def optimal_control(t, lambda_pars, x):
    i, p = x  # State variables
    lambda_1, lambda_2 = lambda_pars
    res = p * k * lambda_(t) - (dynamic_mu(t)) - (lambda_2 * p * k * lambda_(t) - lagrangian(t, x) / i)/ lambda_1
    return -max_down if lambda_1 < lambda_2 else max_up
    #return np.clip(res, -max_down, max_up)
# Define full system with state and costates
def system_with_costates(t, y):
    x = y[:2]  # Extract state variables [i, p]
    lambda_pars = y[2:]  # Extract costate variables [lambda_1, lambda_2]

    # Compute optimal control
    u = optimal_control(t, lambda_pars, x)

    # Compute system dynamics and costate equations
    dx_dt = system_dynamics(t, x, u)
    d_lambda_dt = costate_equations(t, lambda_pars, x, u)

    return dx_dt + d_lambda_dt  # Return full system of ODEs


# Initial conditions
i0 = 0.01  # Initial investors
p0 = 1  # potential investors
lambda1_0 = 0.1 # Initial costate for i
lambda2_0 = 0.1  # Initial costate for p
y0 = [i0, p0, lambda1_0, lambda2_0]

# Time settings
t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)

# Solve system with optimal control
solution_optimal = spi.solve_ivp(system_with_costates, t_span, y0, t_eval=t_eval, method='RK45')

# Extract solutions for optimal control
t = solution_optimal.t
i_opt, p_opt, lambda1_opt, lambda2_opt = solution_optimal.y
u_opt = np.array([optimal_control(t[j], [lambda1_opt[j], lambda2_opt[j]], [i_opt[j], p_opt[j]]) for j in range(len(t))])

# Solve system with fixed control u = -0.2
def system_fixed_u(t, x):
    return system_dynamics(t, x, u=-max_down)

solution_fixed = spi.solve_ivp(system_fixed_u, t_span, y0[:2], t_eval=t_eval, method='RK45')

# Extract solutions for fixed control
i_fixed, p_fixed = solution_fixed.y
u_fixed = np.full_like(t, -max_down)

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, i_opt, label="i(t) (Optimal control)", color="b")
plt.plot(t, i_fixed, label="i(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.axhline(i_0)
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, p_opt, label="p(t) (Optimal control)", color="b")
plt.plot(t, p_fixed, label="p(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, [dynamic_mu(t) + u_opt[i] for i, t in enumerate(t)], label="u(t) (Optimal control)", color="b")
plt.plot(t, [dynamic_mu(t) for i, t in enumerate(t)], label="mu(t) (Market)", color="red")
plt.plot(t, [dynamic_mu(t) + u_fixed[i] for i, t in enumerate(t)], label="u(t) = -0.2 (Fixed)", linestyle="dashed", color="orange")
plt.xlabel("Time")
plt.ylabel("u(t)")
plt.axhline(0.2, color='r', linestyle='--', label="Upper bound 0.2")
plt.legend()

plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots()

# Plot market positivity on the primary y-axis
ax1.plot(t, [interest_calculator.market_positivity(ti) for ti in t], label='Market Positivity', color='b')
ax1.set_xlabel("Time")
ax1.set_ylabel("Market Positivity", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(t, [dynamic_mu(ti) for ti in t], label='μ from Time', color='r')
ax2.set_ylabel("μ (Withdrawal Rate)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

#ax2 = ax1.twinx()
#ax2.plot(t, [interest_calculator.mu_from_market_positivity(interest_calculator.market_positivity(ti)) for ti in t], label='μ from Market Positivity', color='r')


# Add legends
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

plt.title("Market Positivity and μ Over Time")
plt.show()


L_opt = np.array([lagrangian(t[j], [i_opt[j], p_opt[j]]) for j in range(len(t))])
J_opt = trapezoid(L_opt, t)

# Compute integral for fixed control solution
L_fixed = np.array([lagrangian(t[j], [i_fixed[j], p_fixed[j]]) for j in range(len(t))])
J_fixed = trapezoid(L_fixed, t)

print(f"Integral of Lagrangian (Optimal Control): {J_opt:.4f}")
print(f"Integral of Lagrangian (Fixed Control u=-0.2): {J_fixed:.4f}")