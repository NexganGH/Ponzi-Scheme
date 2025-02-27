import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

lambda_ = 0.2
mu = 0.4
k = 5

u_min = 0
u_max = 0.3


# Define the system of differential equations
def system_dynamics(t, y):
    # Extract the state variables (modify according to your equations)
    i, p, lambda1, lambda2 = y
    u = u_max if lambda1 > 0 else u_min

    # Define the derivatives (modify according to your system)
    di_dt = i * p * k * lambda_ - mu * i + u * i  # Example equation for i
    dp_dt = -i * p * k * lambda_  # Example equation for p
    dlambda1_dt = -lambda1 * (p * k * lambda_ - mu * u) + lambda2 * p * k * lambda_  -1# Example eq for λ1
    dlambda2_dt = (lambda2 - lambda1) * i * k * lambda_  # Example eq for λ2

    return [di_dt, dp_dt, dlambda1_dt, dlambda2_dt]


# Boundary value problem solver using shooting method
def bvp_shooting_method(initial_guesses):
    # Initial guesses for λ1(0) and λ2(0)
    lambda1_0, lambda2_0 = initial_guesses

    # Initial conditions for the system
    i0 = 0.1
    p0 = 0.9
    initial_conditions = [i0, p0, lambda1_0, lambda2_0]

    # Define the time span (start, end)
    t_span = (0, 30)  # From time 0 to 30
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time grid for evaluation

    # Solve the IVP
    solution = solve_ivp(system_dynamics, t_span, initial_conditions, t_eval=t_eval)

    # Extract the results from the solution
    lambda1_values, lambda2_values = solution.y[2], solution.y[3]

    # We want lambda1(T) = lambda2(T) = 0
    return [lambda1_values[-1], lambda2_values[-1]]


# Initial guess for λ1(0) and λ2(0)
initial_guesses = [5, 1]

# Use a root-finding method (e.g., Newton's method) to find the correct initial guesses
solution = root(bvp_shooting_method, initial_guesses)

# Extract the optimized initial guesses
lambda1_0_opt, lambda2_0_opt = solution.x

print('found', lambda1_0_opt, lambda2_0_opt)

# Now solve the system with the optimized initial conditions
i0 = 0.1
p0 = 0.9
initial_conditions = [i0, p0, lambda1_0_opt, lambda2_0_opt]
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 100)

solution = solve_ivp(system_dynamics, t_span, initial_conditions, t_eval=t_eval)

# Extract the results from the solution
t_values = solution.t
i_values, p_values, lambda1_values, lambda2_values = solution.y

u_values = np.where(lambda1_values > 0, u_max, u_min)

# Plot the results
fig, ax = plt.subplots(3, 1, figsize=(10, 8))  # 3 plots for i, p, u
ax[0].plot(t_values, i_values, label="i(t)")
ax[0].set_xlabel("Time"), ax[0].set_ylabel("i"), ax[0].legend()
ax[1].plot(t_values, p_values, label="p(t)", color="green")
ax[1].set_xlabel("Time"), ax[1].set_ylabel("p"), ax[1].legend()
ax[2].plot(t_values, u_values, label="u(t)", color="red")
ax[2].set_xlabel("Time"), ax[2].set_ylabel("u"), ax[2].legend()
plt.tight_layout()
plt.show()

# Optionally, plot λ1 and λ2
fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # 2 plots for lambda1 and lambda2
ax[0].plot(t_values, lambda1_values, label="λ1(t)", color="purple")
ax[0].set_xlabel("Time"), ax[0].set_ylabel("λ1"), ax[0].legend()
ax[1].plot(t_values, lambda2_values, label="λ2(t)", color="orange")
ax[1].set_xlabel("Time"), ax[1].set_ylabel("λ2"), ax[1].legend()
plt.tight_layout()
plt.show()
