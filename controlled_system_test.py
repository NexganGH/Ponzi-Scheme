import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the problem parameters
L = 10  # Time horizon
N = 100  # Number of time steps
tgrid = np.linspace(0, L, N)  # Uniform time grid
dt = tgrid[1] - tgrid[0]  # Time step size

# Initial guess for state (x), control (u), and costate (lambda)
x0 = np.ones(N)  # Initial guess for state trajectory
u0 = np.zeros(N)  # Initial guess for control trajectory
lambda0 = np.zeros(N)  # Initial guess for costate trajectory

# Cost function to minimize
def objective(z):
    x = z[:N]  # State trajectory
    u = z[N:2*N]  # Control trajectory
    return np.sum(u**2 + (2 * x * np.exp(-x**2))**2) * dt  # Integral approximation

# System dynamics constraint using finite differences
def dynamics_constraint(z):
    x = z[:N]
    u = z[N:2*N]
    dx_dt = np.zeros(N)
    dx_dt[:-1] = (x[1:] - x[:-1]) / dt  # Forward Euler approximation
    f_x = -2 * x * np.exp(-x**2) + u  # System dynamics
    return dx_dt[:-1] - f_x[:-1]  # Must be zero for all time steps

# Costate equation (Pontryagin's condition)
def costate_constraint(z):
    x = z[:N]
    lam = z[2*N:]
    dlam_dt = np.zeros(N)
    dlam_dt[:-1] = (lam[1:] - lam[:-1]) / dt  # Forward Euler approximation
    dH_dx = -4 * x * np.exp(-x**2) * (1 - lam * np.exp(-x**2))  # Costate equation
    return dlam_dt[:-1] - dH_dx[:-1]  # Must be zero for all time steps

# Boundary conditions
def boundary_conditions(z):
    x = z[:N]
    u = z[N:2*N]
    lam = z[2*N:]
    return [x[0] - 1, u[0], lam[-1]]  # x(0) = 1, u(0) = 0, lambda(L) = 0

# Combine constraints
constraints = [
    {"type": "eq", "fun": dynamics_constraint},
    {"type": "eq", "fun": costate_constraint},
    {"type": "eq", "fun": boundary_conditions}
]

# Define bounds for control variable u(t) between 0.5 and 1
bounds = [(None, None)] * N + [(0.5, 1)] * N + [(None, None)] * N  # Bounds on u(t)

# Initial guess for optimization
z0 = np.concatenate([x0, u0, lambda0])

# Solve the optimization problem with bounds
solution = minimize(objective, z0, constraints=constraints, bounds=bounds, method="SLSQP")

# Extract optimal solutions
x_opt = solution.x[:N]
u_opt = solution.x[N:2*N]
lambda_opt = solution.x[2*N:]

# Plot results
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(tgrid, x_opt, label="x(t)")
plt.xlabel("t"), plt.ylabel("x"), plt.legend()

plt.subplot(1, 3, 2)
plt.plot(tgrid, u_opt, label="u(t)", color="red")
plt.xlabel("t"), plt.ylabel("u"), plt.legend()

plt.subplot(1, 3, 3)
plt.plot(tgrid, lambda_opt, label="λ(t)", color="green")
plt.xlabel("t"), plt.ylabel("λ"), plt.legend()

plt.show()
