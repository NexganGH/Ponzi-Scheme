import numpy as np
import scipy.integrate as spi
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

#def diffusion_rate(i, p, lambda_1, lambda_2, t):
    #return np.where(lambda_1 > lambda_2, lambda_max, lambda_min)
 #   return np.clip((lambda_1 * dynamic_mu(t) * i - lagrangian(i, p, t))/(p*k*i * (lambda_1 - lambda_2)), lambda_min, lambda_max)
# Extended system dynamics with costates
def system_dynamics(t, x):
    i, p, lambda_1, lambda_2 = x  # Include the costates in the state vector
    mu = dynamic_mu(t) + np.where(lambda_1 > lambda_2, mu_)
    lambda_fixed = diffusion_rate(i, p, lambda_1, lambda_2, t)
    di_dt = p * lambda_fixed * k * i - mu * i
    dp_dt = -p * lambda_fixed * k * i
    dlambda1_dt = -lambda_1 * (p * k * lambda_fixed - dynamic_mu(t)) + lambda_2 * p * k * lambda_fixed - dL_di(t, x)
    dlambda2_dt = lambda_1 * k * lambda_fixed * i - lambda_2 * k * lambda_fixed * i
    return np.array([di_dt, dp_dt, dlambda1_dt, dlambda2_dt])

# Boundary value problem dynamics
def system_bvp(t, y):
    return system_dynamics(t, y)

# Boundary conditions for i, p, lambda_1, lambda_2
def bc(ya, yb):
    return np.array([
        ya[0] - i0,          # Boundary condition for i(0)
        ya[1] - p0,          # Boundary condition for p(0)
        yb[2] - 1,           # Boundary condition for lambda_1(T)
        yb[3] - 0            # Boundary condition for lambda_2(T)
    ])

# Initial conditions for i, p, lambda_1, lambda_2
i0 = 0.1
p0 = 1 - i0
lambda_1_0 = 0  # Initial value of lambda_1
lambda_2_0 = 0  # Initial value of lambda_2
y0 = [i0, p0, lambda_1_0, lambda_2_0]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 2000)

# First, solve the system for an unconstrained problem to get a better guess
initial_guess_solution = spi.solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval)

# Get the trajectory from the unconstrained system
i_guess = initial_guess_solution.y[0]
p_guess = initial_guess_solution.y[1]
lambda_1_guess = initial_guess_solution.y[2]
lambda_2_guess = initial_guess_solution.y[3]
y_guess = np.vstack([i_guess, p_guess, lambda_1_guess, lambda_2_guess])

# Now solve the boundary value problem using the improved guess
solution = spi.solve_bvp(system_bvp, bc, t_eval, y_guess, max_nodes=5000)

print(solution.message)
t = solution.x
i_opt, p_opt, lambda_1_opt, lambda_2_opt = solution.y

# Calculate the diffusion_rate over time
diff_rate_opt = diffusion_rate(i_opt, p_opt, lambda_1_opt, lambda_2_opt, t)

plt.figure(figsize=(10, 8))

# Plot i(t)
plt.subplot(5, 1, 1)
plt.plot(t, i_opt, label="i(t) (Lambda fixed)", color="b")
plt.plot(initial_guess_solution.t, i_guess, label='Initial guess')
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

# Plot p(t)
plt.subplot(5, 1, 2)
plt.plot(t, p_opt, label="p(t) (Lambda fixed)", color="r")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

# Plot lambda_1(t)
plt.subplot(5, 1, 3)
plt.plot(t, lambda_1_opt, label="lambda_1(t)", color="g")
plt.xlabel("Time")
plt.ylabel("lambda_1(t)")
plt.legend()

# Plot lambda_2(t)
plt.subplot(5, 1, 4)
plt.plot(t, lambda_2_opt, label="lambda_2(t)", color="orange")
plt.xlabel("Time")
plt.ylabel("lambda_2(t)")
plt.legend()

# Plot diffusion_rate(t)
plt.subplot(5, 1, 5)
plt.plot(t, diff_rate_opt, label="Diffusion Rate", color="purple")
plt.xlabel("Time")
plt.ylabel("Diffusion Rate")
plt.legend()

plt.tight_layout()
plt.show()
