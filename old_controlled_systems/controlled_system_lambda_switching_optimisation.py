import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# System parameters
k = 5
lambda_fixed = 0.2  # Assumed constant lambda
lambda_min = 0.1
lambda_max = 0.5


def dynamic_mu(ti):
    return np.where(ti < 17.5, 0.15, np.where(ti <= 20, 0.7, 0.15))


def dL_di(t, x):
    return -1


def lagrangian(i, p, t):
    return i


def diffusion_rate(i, p, lambda_1, lambda_2, t):
    return np.where(lambda_1 > lambda_2, lambda_max, lambda_min)


# System dynamics with costates
def system_dynamics(t, x):
    i, p, lambda_1, lambda_2 = x  # Include the costates in the state vector
    mu = dynamic_mu(t)
    lambda_fixed = diffusion_rate(i, p, lambda_1, lambda_2, t)
    di_dt = p * lambda_fixed * k * i - mu * i
    dp_dt = -p * lambda_fixed * k * i
    dlambda1_dt = -lambda_1 * (p * k * lambda_fixed - dynamic_mu(t)) + lambda_2 * p * k * lambda_fixed - dL_di(t, x)
    dlambda2_dt = lambda_1 * k * lambda_fixed * i - lambda_2 * k * lambda_fixed * i
    return np.array([di_dt, dp_dt, dlambda1_dt, dlambda2_dt])


# Objective function to minimize
def objective(z):
    N = len(z) // 4
    i = z[:N]  # State trajectory i(t)
    p = z[N:2 * N]  # State trajectory p(t)
    lambda_1 = z[2 * N:3 * N]  # Costate trajectory lambda_1(t)
    lambda_2 = z[3 * N:]  # Costate trajectory lambda_2(t)

    # Compute the diffusion rate and the Lagrangian for each time step
    diff_rate = diffusion_rate(i, p, lambda_1, lambda_2, tgrid)
    total_diffusion = np.sum(i)  # Summing the diffusion rate as the cost to minimize

    # You can also add other terms to the cost function depending on your objective
    return total_diffusion


# Constraint functions (system dynamics and boundary conditions)
def constraint(z):
    N = len(z) // 4
    i = z[:N]
    p = z[N:2 * N]
    lambda_1 = z[2 * N:3 * N]
    lambda_2 = z[3 * N:]

    # Initialize the derivatives
    dx_dt = np.zeros_like(i)

    # System dynamics constraints
    for idx in range(1, N):
        dx_dt[idx - 1] = i[idx] - \
                         system_dynamics(tgrid[idx], [i[idx - 1], p[idx - 1], lambda_1[idx - 1], lambda_2[idx - 1]])[
                             0]  # i(t)

    # Boundary conditions for i(0), p(0), lambda_1(T), lambda_2(T)
    bc = np.array([i[0] - i0, p[0] - p0, lambda_1[-1] - 1, lambda_2[-1]])

    return np.concatenate([dx_dt, bc])


# Time grid and initial conditions
L = 30  # Time horizon
N = 100  # Number of time steps
tgrid = np.linspace(0, L, N)

# Initial conditions for i, p, lambda_1, lambda_2
i0 = 0.1
p0 = 1 - i0
lambda_1_0 = 1
lambda_2_0 = 0.5
initial_guess = np.concatenate([np.ones(N) * i0, np.ones(N) * p0, np.zeros(N), np.zeros(N)])

# Bounds (if any specific bounds on the states/controls are needed)
bounds = [(None, None)] * N + [(None, None)] * N + [(None, None)] * N + [
    (None, None)] * N  # No bounds on state or costates

# Solve optimization problem
result = minimize(objective, initial_guess, constraints={"type": "eq", "fun": constraint}, bounds=bounds,
                  method="SLSQP")

# Extract the optimized results
i_opt = result.x[:N]
p_opt = result.x[N:2 * N]
lambda_1_opt = result.x[2 * N:3 * N]
lambda_2_opt = result.x[3 * N:]

# Plot results
plt.figure(figsize=(10, 8))

# Plot i(t)
plt.subplot(5, 1, 1)
plt.plot(tgrid, i_opt, label="i(t) (Optimized)")
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

# Plot p(t)
plt.subplot(5, 1, 2)
plt.plot(tgrid, p_opt, label="p(t) (Optimized)")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

# Plot lambda_1(t)
plt.subplot(5, 1, 3)
plt.plot(tgrid, lambda_1_opt, label="lambda_1(t) (Optimized)", color="green")
plt.xlabel("Time")
plt.ylabel("lambda_1(t)")
plt.legend()

# Plot lambda_2(t)
plt.subplot(5, 1, 4)
plt.plot(tgrid, lambda_2_opt, label="lambda_2(t) (Optimized)", color="orange")
plt.xlabel("Time")
plt.ylabel("lambda_2(t)")
plt.legend()

# Plot diffusion_rate(t)
diff_rate_opt = diffusion_rate(i_opt, p_opt, lambda_1_opt, lambda_2_opt, tgrid)
plt.subplot(5, 1, 5)
plt.plot(tgrid, diff_rate_opt, label="Diffusion Rate", color="purple")
plt.xlabel("Time")
plt.ylabel("Diffusion Rate")
plt.legend()

plt.tight_layout()
plt.show()
