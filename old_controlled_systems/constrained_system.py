import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Define the time span and grid
T_start, T_end = 0, 30  # Time range
N = 100  # Number of time steps
t_grid = np.linspace(T_start, T_end, N)  # Time discretization

# Define initial guesses
i0 = np.zeros(N) * 0.1  # Initial guess for i
p0 = np.ones(N)  # Initial guess for p
u0 = np.ones(N)  # Initial guess for u
lambda1_0 = np.ones(N)  # Initial guess for λ1
lambda2_0 = -np.ones(N)  # Initial guess for λ2
lambda_ = 0.2
mu = 0.4
k = 5
u_min, u_max = 0., 0.3  # Adjust as needed
bounds = [(0, 1)] * N + [(0, 1)] * N + [(u_min, u_max)] * N + [(None, None)] * N + [(None, None)] * N

i_min=0.2
def lagrangian(i, p, u, lambda1, lambda2):
    #return i
    return np.where(i>i_min, 0, 1)
# Define the Hamiltonian function
def hamiltonian(z):
    i, p, u, lambda1, lambda2 = np.split(z, 5)  # Split the optimization variables
    H = lambda1 * (p * k * lambda_ - mu + u) * i - lambda2 * (i * p * k * lambda_)  + lagrangian(i,p,u,lambda1,lambda2)#(some_function(i, p, u, lambda1, lambda2))  # Define the function you want to minimize
    return np.sum(lagrangian(i, p,u, lambda1, lambda2)) * (t_grid[1] - t_grid[0])  # Sum over all time steps for discretization


# Define the system of differential equations
def system_dynamics(t, y):
    i, p, u, lambda1, lambda2 = y  # Extract variables
    di_dt = i * p * k * lambda_ - mu * i + u * i
    dp_dt = -i * p * k * lambda_  # Define the derivative of p
    #du_dt =   # Define the derivative of u
    dlambda1_dt = - lambda1*(p * k * lambda_ - mu * u) + lambda2*p*k*lambda_  # Define λ1 evolution
    dlambda2_dt = lambda2*p*k*lambda_  # Define λ2 evolution
    return [di_dt, dp_dt, dlambda1_dt, dlambda2_dt]


# Define constraints (boundary conditions and dynamic constraints)
def constraints(z):
    i, p, u, lambda1, lambda2 = np.split(z, 5)
    constraints_list = []

    # Boundary conditions
    constraints_list.append(i[0] - 0.01)  # Example boundary condition for i
    constraints_list.append(p[0] - 0.99)  # Example boundary condition for p
    #constraints_list.append(u[0] - some_initial_value)  # Example boundary condition for u
    constraints_list.append(lambda1[-1] - 0)  # Example final condition for λ1
    constraints_list.append(lambda2[-1] - 0)  # Example final condition for λ2

    # System dynamics constraint (discretized version)
    dt = t_grid[1] - t_grid[0]
    for k in range(N - 1):
        constraints_list.append(
            (i[k + 1] - i[k]) / dt - system_dynamics(t_grid[k], [i[k], p[k], u[k], lambda1[k], lambda2[k]])[0])
        constraints_list.append(
            (p[k + 1] - p[k]) / dt - system_dynamics(t_grid[k], [i[k], p[k], u[k], lambda1[k], lambda2[k]])[1])
        constraints_list.append(
            (lambda1[k + 1] - lambda1[k]) / dt - system_dynamics(t_grid[k], [i[k], p[k], u[k], lambda1[k], lambda2[k]])[
                2])
        constraints_list.append(
            (lambda2[k + 1] - lambda2[k]) / dt - system_dynamics(t_grid[k], [i[k], p[k], u[k], lambda1[k], lambda2[k]])[
                3])

    return np.array(constraints_list)


fig, axes = plt.subplots(2, 1, figsize=(8, 6))
ax1, ax2 = axes
(line1,) = ax1.ba_plot(t_grid, np.zeros_like(t_grid), label="i(t)")
(line2,) = ax1.ba_plot(t_grid, np.zeros_like(t_grid), label="p(t)")
ax1.legend()
ax1.set_title("State Variables (i, p)")
ax1.set_xlabel("t")
ax1.set_ylabel("Value")

(line3,) = ax2.ba_plot(t_grid, np.zeros_like(t_grid), label="u(t)", color="red")
ax2.legend()
ax2.set_title("Control Variable u")
ax2.set_xlabel("t")
ax2.set_ylabel("Value")

plt.ion()  # Enable interactive mode
plt.show(block=False)  # Show figure without blocking

# Callback function for live update
def callback(z):
    i, p, u, lambda1, lambda2 = np.split(z, 5)

    # Update plot data
    line1.set_ydata(i)
    line2.set_ydata(p)
    line3.set_ydata(u)

    # Auto-scale axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    # Force redraw
    fig.canvas.draw()
    fig.canvas.flush_events()
    #i, p, u, lambda1, lambda2 = np.split(z, 5)
    H_value = hamiltonian(z)
    print(f"Hamiltonian: {H_value:.4f}")
    print(f"Cost is: {lagrangian(i, p, u, lambda1, lambda2).sum() * (t_grid[2] - t_grid[1])}")

# Initial guess for optimization
z0 = np.concatenate([i0, p0, u0, lambda1_0, lambda2_0])

# Solve optimization problem
solution = minimize(hamiltonian, z0, constraints={'type': 'eq', 'fun': constraints}, bounds=bounds, method="SLSQP", options={'disp': True}, callback=callback)
# Extract optimal variables
i_opt, p_opt, u_opt, lambda1_opt, lambda2_opt = np.split(solution.x, 5)
plt.ioff()
# Plot results
fig, ax = plt.subplots(3, 1, figsize=(10, 8))  # 3 plots for i, p, u
ax[0].ba_plot(t_grid, i_opt, label="i(t)")
ax[0].set_xlabel("Time"), ax[0].set_ylabel("i"), ax[0].legend()
ax[1].ba_plot(t_grid, p_opt, label="p(t)", color="green")
ax[1].set_xlabel("Time"), ax[1].set_ylabel("p"), ax[1].legend()
ax[2].ba_plot(t_grid, u_opt, label="u(t)", color="red")
ax[2].set_xlabel("Time"), ax[2].set_ylabel("u"), ax[2].legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # 2 plots for lambda1 and lambda2
ax[0].ba_plot(t_grid, lambda1_opt, label="λ1(t)", color="purple")
ax[0].set_xlabel("Time"), ax[0].set_ylabel("λ1"), ax[0].legend()
ax[1].ba_plot(t_grid, lambda2_opt, label="λ2(t)", color="orange")
ax[1].set_xlabel("Time"), ax[1].set_ylabel("λ2"), ax[1].legend()
plt.tight_layout()
plt.show(block=True)
