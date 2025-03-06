import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from simulation.finance_data import FinanceData
from simulation.parameters_calculator import InterestCalculator
from scipy.optimize import fsolve

data = FinanceData()
data.download()
interest_calculator = InterestCalculator(rp=lambda t:0.1, rr=data.interpolated_r_r)
interest_calculator.compute_market_positivity(0, 30, 100)
#dynamic_mu = interest_calculator.mu_from_rr_func(max=0.5, steepness=30)
dynamic_mu = lambda t: 0.2
# System parameters
#lambda_ = 0.2
k = 5

rr = lambda t: 0.1#interest_calculator.r_r
rp = lambda t: 0.1#interest_calculator.r_p
M = 100
N = 10000
# suppose the control is on the spreading rate, which must be between 0 and 0.25

# Define system dynamics
def system_dynamics(t, x, u):
    i, p, S, U, C = x  # State variables
    mu = u#dynamic_mu(t)  # Withdrawal rate
    lambda_ = 0.25# u

    di_dt = p * lambda_ * k * i - mu * i
    dp_dt = -p * lambda_ * k * i
    avg_w = U / C if C != 0 else 0
    dS_dt = ((S * rr(t)
              + M * N * lambda_ * p * k * i)
             - N * mu * i * avg_w)
    #print('evaluating system dynamics with ', x, ' and u', u)
    dU_dt = lambda_ * k * p * i * M + (rp(t) - mu) * U
    dC_dt = lambda_ * k * p * i - mu * C


    return [di_dt, dp_dt, dS_dt, dU_dt, dC_dt]


# Define costate equations
def costate_equations(t, costates, x, u):
    i, p, S, U, C= x  # State variables
    l1, l2, l3, l4, l5 = costates  # Costate variables
    lambda_ = 0.25#u
    mu = u#dynamic_mu(t)
    # Compute derivatives of costates
    avg_w = U / C if C != 0 else 0
    dlambda1_dt = - ( l1 * (p * k * lambda_ - mu) - l2 * p * k * lambda_ + l3 * (M * N * lambda_ * p * k - N * mu * avg_w) + l4 * (lambda_ * k * p * M) + l5 * (lambda_*k*p))
    dlambda2_dt = - (l1 * (lambda_ * k * i) - l2 * lambda_ * k * i + l3 * (M * N * lambda_ * k * i) + l4 * (M*N*lambda_*k*i) + l5 * (lambda_*k*i) )
    dlambda3_dt = - l3 * rr(t)
    dlambda4_dt = (0 if C == 0 else l3 * N * mu * i / C) - l4*(rp(t) - mu)
    dlambda5_dt = (0 if C**2 == 0 else - (l3 * (N * mu * i * U)/C**2) - mu * l5)
    return [dlambda1_dt, dlambda2_dt, dlambda3_dt, dlambda4_dt, dlambda5_dt]


# Define optimal control function with constraints
#def optimal_control(lambda_1, i):
 #   u_unconstrained = -lambda_1 * i  # Derived from dH/du = 0
  #  u_min, u_max = -0.2, 0.2  # Control bounds
   # return np.clip(u_unconstrained, u_min, u_max)  # Apply bounds
def lagrangian(t, x, u):
    i, p, S, U, C= x
    gamma = 20
    penalty_term = np.exp(gamma * (u - 0.3)**2)

    if S <= M * 10:
        return 1 + penalty_term
    return penalty_term

def hamiltonian_function(t, costates, x):

    #mu = dynamic_mu(t)
    i, p, S, U, C = x
    l1, l2, l3, l4, l5 = costates
    avg_W = U / C if C != 0 else 0
    #print('evaluating hamiltonian, we have', i, p, S, U, C, l1, l2, l3, l4, l5, avg_W)
    def hamiltonian(u):
        #print(l1 * (u * p * k * i - mu * i), ' ', l2 * (-u * p * k * i), ' ', l3 * (S * rr(t) + M * N * U * p * k * i - N*mu*i*avg_W), ' ', l4 * (u * k * p * i * M + (rp(t) - mu)*U), ' ', l5*(u*k*p*i-mu*C), ' ', lagrangian(t, x))
        #print('u is ', u)
        lambda_ = 0.2
        mu = u
        return l1 * (lambda_ * p * k * i - mu * i) + l2 * (-lambda_ * p * k * i) + l3 * (S * rr(t) + M * N * lambda_ * p * k * i - N * mu * i * avg_W) + l4 * (lambda_ * k * p * i * M + (rp(t) - mu) * U) + l5*(lambda_ * k * p * i - mu * C) + lagrangian(t, x, lambda_)
    return lambda u: hamiltonian(u[0])

def optimal_control(t, costates, x):
    i, p, S, U, C = x  # State variables
    hamiltonian = hamiltonian_function(t, costates, x)
    #res = p * k * lambda_ - (0.25) - (lambda_2 * p * k * lambda_ - lagrangian(t, x) / i)/ lambda_1

    res = fsolve(hamiltonian, np.array(0.3))
    print(f'the optimal u found is {res[0]} with hamiltonian {hamiltonian(res)}')
    #print('solving hamiltonian yielded values of ', res)
    #return np.clip(res[0], 0, 0.3)
    return res[0]
    #return np.clip(res[0], 0, 0.3)
# Define full system with state and costates
def system_with_costates(t, y):
    x = y[:5]  # Extract state variables [i, p]
    lambda_pars = y[5:]  # Extract costate variables [lambda_1, lambda_2]

#    print(x, lambda_pars)
    # Compute optimal control
    u = optimal_control(t, lambda_pars, x)

    # Compute system dynamics and costate equations
    dx_dt = system_dynamics(t, x, u)
    d_lambda_dt = costate_equations(t, lambda_pars, x, u)
 #   print(dx_dt + d_lambda_dt)

    return dx_dt + d_lambda_dt  # Return full system of ODEs


# Initial conditions
i0 = 0.01  # Initial investors
p0 = 1  # potential investors
lambda1_0 = 0.1 # Initial costate for i
lambda2_0 = 0.1  # Initial costate for p
lambda3_0 = 0.1 # Initial costate for i
lambda4_0 = 0.1
lambda5_0 = 0.1 # Initial costate for i
y0 = [1/N, 1, 0, 0, 0, lambda1_0, lambda2_0, lambda3_0, lambda4_0, lambda5_0]

# Time settings
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Solve system with optimal control
solution_optimal = spi.solve_ivp(system_with_costates, t_span, y0, t_eval=t_eval, method='RK45')

# Extract solutions for optimal control
t = solution_optimal.t
sol_opt = solution_optimal.y
print(solution_optimal.message)
print('ended evaluation, and sol_opt is', sol_opt)

u_opt = np.array([optimal_control(t[j], sol_opt[5:, j], sol_opt[:5, j]) for j in range(len(t))])

# Solve system with fixed control u = -0.2
def system_fixed_u(t, x):
    return system_dynamics(t, x, u=0.3)

solution_fixed = spi.solve_ivp(system_fixed_u, t_span, y0[:5], t_eval=t_eval, method='RK45')

# Extract solutions for fixed control
sol_fixed = solution_fixed.y
u_fixed = np.full_like(t, 0.3)

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, sol_opt[0], label="i(t) (Optimal control)", color="b")
plt.plot(t, sol_fixed[0], label="i(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.axhline(0.4)
plt.xlabel("Time")
plt.ylabel("i(t)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, sol_opt[1], label="p(t) (Optimal control)", color="b")
plt.plot(t, sol_fixed[1], label="p(t) (Fixed control u=-0.2)", linestyle="dashed", color="orange")
plt.xlabel("Time")
plt.ylabel("p(t)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, [u_opt[i] for i, t in enumerate(t)], label="u(t) (Optimal control)", color="b")
plt.plot(t, [u_fixed[i] for i, t in enumerate(t)], label="u(t) = -0.2 (Fixed)", linestyle="dashed", color="orange")
plt.xlabel("Time")
plt.ylabel("u(t)")
plt.axhline(-0.2, color='r', linestyle='--', label="Lower bound -0.2")
plt.axhline(0.2, color='r', linestyle='--', label="Upper bound 0.2")
plt.ylim((0, 0.5))
plt.legend()

plt.tight_layout()
plt.show()



L_opt = np.array([lagrangian(t[j], sol_opt[:5, j], u_opt[j]) for j in range(len(t))])
J_opt = trapezoid(L_opt, t)

# Compute integral for fixed control solution
L_fixed = np.array([lagrangian(t[j], sol_fixed[:5, j], u_fixed[j]) for j in range(len(t))])
J_fixed = trapezoid(L_fixed, t)

print(f"Integral of Lagrangian (Optimal Control): {J_opt:.4f}")
print(f"Integral of Lagrangian (Fixed Control u=-0.2): {J_fixed:.4f}")
