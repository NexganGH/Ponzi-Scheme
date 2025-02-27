import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpmath import sigmoid
from scipy.integrate import trapezoid
from data import Data
from networks.interest_calculator import InterestCalculator

#data = Data()
#data.download()
#interest_calculator = InterestCalculator(r_p=lambda t:0.1, r_r=data.interpolated_r_r)
#interest_calculator.compute_market_positivity(0, 30, 100)
#dynamic_mu = #interest_calculator.mu_from_rr_func(min=0.05, base=0.1, max=0.5, steepness=100)
def dynamic_mu(ti):
    if ti < 17.5:
        return 0.15
    elif 17.5 <= ti <= 20:
        return 0.7
    else:
        return 0.15
# System parameters
k = 5

min_lambda=0
max_lambda=0.4

def compute_lambda(u):
    return np.clip(u, min_lambda, max_lambda)
    #return min_lambda + max_lambda*sigmoid(u)

# Define system dynamics
def system_dynamics(t, x, u):
    i, p = x  # State variables
    mu = dynamic_mu(t)  # Withdrawal rate

    lambda_ = compute_lambda(u)
    di_dt = p * lambda_ * k * i - mu * i
    dp_dt = -p * lambda_ * k * i

    return [di_dt, dp_dt]

i_0 = 0.2
def dL_di(i):
    return 0
    if np.abs(i - i_0) < 0.03:
        return 500
    else:
        return 0

# Define costate equations
def costate_equations(t, lambda_pars, x, u):
    i, p = x  # State variables
    lambda_1, lambda_2 = lambda_pars  # Costate variables

    # Compute derivatives of costates
    lambda_ = min_lambda + sigmoid(u)
    dlambda1_dt = lambda_1 * (p * k * lambda_ - dynamic_mu(t)) - lambda_2 * p * k * lambda_ + dL_di(i)#-lambda_1 * (lambda_p * lambda_ * k - (0.25 + u))
    dlambda2_dt = lambda_1 * k * lambda_ * i - lambda_2 * k * lambda_ * i

    return [-dlambda1_dt*0, -dlambda2_dt*0]


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
    #res = p * k * lambda_(t) - (dynamic_mu(t)) - (lambda_2 * p * k * lambda_(t) - lagrangian(t, x) / i)/ lambda_1

    #res = (lagrangian(t, x)/i + dynamic_mu(t) * lambda_1)/(p*k*(lambda_1 - lambda_2))
    #res = (dynamic_mu(t) * lambda_2 - lagrangian(t, x) / i)/(k*p*(lambda_1 - lambda_2))
    switching_func = lambda_1 - lambda_2
    return max_lambda
    return max_lambda if switching_func > 0 else min_lambda
    #return np.clip(res, min_lambda, max_lambda)
    #print('calculated ', res, compute_lambda(res))
    #return min(max_lambda, max(res, min_lambda))
# Define full system with state and costates
def system_with_costates(t, y):
    num_points = t.shape[0]  # Get the number of time points
    dy_dt = np.zeros_like(y)  # Initialize derivative array with the same shape as y

    for i in range(num_points):
        x = y[:2, i]  # Extract state variables [i, p] at time t[i]
        lambda_pars = y[2:, i]  # Extract costate variables [lambda_1, lambda_2] at time t[i]

        # Compute optimal control for this specific time step
        u = optimal_control(t[i], lambda_pars, x)

        # Compute system dynamics and costate equations
        dx_dt = system_dynamics(t[i], x, u)
        d_lambda_dt = costate_equations(t[i], lambda_pars, x, u)

        # Store derivatives in dy_dt
        dy_dt[:2, i] = dx_dt  # Store [di/dt, dp/dt]
        dy_dt[2:, i] = d_lambda_dt  # Store [dlambda_1/dt, dlambda_2/dt]

    return dy_dt


# Initial conditions
i0 = i_0  # Initial investors
p0 = 1-i0  # potential investors
lambda1_0 = 1 # Initial costate for i
lambda2_0 = 0.1  # Initial costate for p
y0 = [i0, p0, lambda1_0, lambda2_0]

# Time settings
t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)

# Solve system with optimal control

def bc(ya, yb):
    #i0, p0 = , 1  # Initial conditions
    lambda1_T, lambda2_T = 0, 0  # Final conditions
    return np.array([
        ya[0] - i0,  # i(0) = i0
        ya[1] - p0,  # p(0) = p0
        yb[2] - lambda1_T-1,  # lambda_1(T) = lambda1_T
        yb[3] - lambda2_T   # lambda_2(T) = lambda2_T
    ])

#i_guess = np.linspace(0.3, 0.3, len(t_eval))  # Assume i(t) is constant initially
# i_guess = np.linspace(0.4, 4, len(t_eval))#0.01 + (0.5 - 0.01) * np.exp(t_eval/30)
# p_guess = p0 * np.exp(t_eval/30)#np.linspace(p0, p0, len(t_eval))  # Assume p(t) is constant initially
# lambda1_guess = np.linspace(1, 1, len(t_eval))  # Assume lambda_1 is constant
# lambda2_guess = np.linspace(0, 0, len(t_eval))  # Assume lambda_2 is constant

# Stack guesses into a (4, num_points) array
y_guess = np.vstack([i_guess, p_guess, lambda1_guess, lambda2_guess])

solution_optimal = spi.solve_bvp(system_with_costates, bc, t_eval, y_guess)
print("Solver success:", solution_optimal.success)
print("Solver message:", solution_optimal.message)
# Extract solutions for optimal control
t = solution_optimal.x
i_opt, p_opt, lambda1_opt, lambda2_opt = solution_optimal.y
u_opt = np.array([optimal_control(t[j], [lambda1_opt[j], lambda2_opt[j]], [i_opt[j], p_opt[j]]) for j in range(len(t))])

# Solve system with fixed control u = -0.2
def system_fixed_u(t, x):
    return system_dynamics(t, x, max_lambda)

solution_fixed = spi.solve_ivp(system_fixed_u, t_span, y0[:2], t_eval=t_eval, method='RK45')

# Extract solutions for fixed control
i_fixed, p_fixed = solution_fixed.y
u_fixed = np.full_like(t, max_lambda)

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, i_opt, label="i(t) (Optimal control)", color="b")
plt.plot(solution_fixed.t, i_fixed, label="i(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.axhline(i_0)
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, p_opt, label="p(t) (Optimal control)", color="b")
plt.plot(solution_fixed.t, p_fixed, label="p(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, [compute_lambda(u_opt[i]) for i, t in enumerate(t)], label="u(t) (Optimal control)", color="b")
#plt.plot(t, [dynamic_mu(t) for i, t in enumerate(t)], label="mu(t) (Market)", color="red")
plt.plot(solution_fixed.t, [u_fixed[i] for i, t in enumerate(solution_fixed.t)], label="u(t) = -0.2 (Fixed)", linestyle="dashed", color="orange")
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
L_fixed = np.array([lagrangian(t[j], [i_fixed[j], p_fixed[j]]) for j in range(len(solution_fixed.t))])
J_fixed = trapezoid(L_fixed, solution_fixed.t)

print(f"Integral of Lagrangian (Optimal Control): {J_opt:.4f}")
print(f"Integral of Lagrangian (Fixed Control u=-0.2): {J_fixed:.4f}")