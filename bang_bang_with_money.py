import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from data import Data
from pyswarm import pso  # Install with `pip install pyswarm`
from networks.interest_calculator import InterestCalculator
from scipy.optimize import differential_evolution


data = Data()
data.download()

interest_calculator = InterestCalculator(r_p = lambda t: 0.2, r_r = data.interpolated_r_r)#lambda t: data.interpolated_r_r(t))#data.interpolated_r_r)
interest_calculator.compute_market_positivity(0, 30, 100)

#lambda_ = lambda _: 0.03#interest_calculator.lambda_from_rr_func(base=0.2, min=0.05, max=0.2, steepness=30)


#rp = interest_calculator.r_p
def mu(t):
    #return 0.05
    return interest_calculator.mu_from_rr_func(base=0.025, min=0.02, max=0.8, steepness=150)(t)

u_min, u_max = 0, 0.25
def lambda_(rp):
    return rp / u_max * 0.035
    #np.where(0.03 * rp(t), )

M = 100  # Define constants if necessary
N = 10000
avg_k = 10




def system_dynamics(t, y, u):

    i, p, S, U, C = y
    rp = u
    lambda_eval = lambda_(rp)
    #print(f'at {t} lambda is {lambda_eval}, mu is {mu(t)}')
    #print(f'pars: {i}, {p}, {S}, {U}, {C}, {N}, {M}, {avg_k}, {interest_calculator.r_p(t)}, {interest_calculator.r_r(t)}')
    mu_eval = mu(t)
    avg_w = U / C if C != 0 else 0

    di_dt = i * p * avg_k * lambda_eval - mu_eval * i
    dp_dt = -i * p * avg_k * lambda_eval
    dS_dt = (S * interest_calculator.r_r(t) + M * N * lambda_eval * p * avg_k * i) - N * mu_eval * i * avg_w
    dU_dt = lambda_eval * avg_k * p * i * M + (rp - mu_eval) * U
    dC_dt = lambda_eval * avg_k * p * i - mu_eval * C

    return [di_dt, dp_dt, dS_dt, dU_dt, dC_dt]


def optimal_control(switching_times):
    def control_function(t):
        switching_times_sorted = sorted(switching_times)
        t = np.array(t)
        control_values = np.zeros_like(t)

        for i in range(len(switching_times_sorted) - 1):
            if i % 2 == 1:
                control_values[(t >= switching_times_sorted[i]) & (t < switching_times_sorted[i + 1])] = u_min
            else:
                control_values[(t >= switching_times_sorted[i]) & (t < switching_times_sorted[i + 1])] = u_max

        if len(switching_times_sorted) % 2 == 1:
            control_values[t >= switching_times_sorted[-1]] = u_max

        return control_values

    return control_function


i0, p0, S0, U0, C0 = 1./N, 1 - 1./N, 5000, 0, 0
t_span = np.linspace(0, 30, 100)
print([interest_calculator.r_r(ti) for ti in t_span])

S_min = 10000
i_threshold = 0.1
def lagrangian(y):
    i,p,S,U,C = y
    return np.where((S > S_min) & (i > i_threshold), 0, 1) - S
def J(switching_times):
    u = optimal_control(switching_times)
    sol = solve_ivp(lambda t, y: system_dynamics(t, y, u(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)
    cost = np.sum(lagrangian(sol.y)) * (t_span[1] - t_span[0])

    #print('tried ', switching_times, ' with cost ', cost)
    return cost


initial_guess = np.zeros(5)#np.linspace(0, 30, 2)
# sol = minimize(lambda switching_times: J(switching_times),
#                initial_guess,
#                bounds=[(0, 30)] * len(initial_guess),
#                tol=1e-10,
#                options={'maxiter': 500, 'xtol': 1e-2},
#                method='Powell')
# bounds = [(0, 30)] * 10#len(initial_guess)
# sol = differential_evolution(J, bounds, strategy='best1bin', tol=1e-6, maxiter=200)
# opt_switching_times = sol.x
#
lb = [0] * len(initial_guess)   # Lower bounds
ub = [30] * len(initial_guess)  # Upper bounds

# Run PSO
opt_switching_times, best_cost = pso(J, lb, ub, swarmsize=20, maxiter=200)

u_opt = optimal_control(opt_switching_times)
sol_opt = solve_ivp(lambda t, y: system_dynamics(t, y, u_opt(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)
i_opt, p_opt, S_opt, U_opt, C_opt = sol_opt.y

u_always_max = optimal_control([0])
sol_max = solve_ivp(lambda t, y: system_dynamics(t, y, u_always_max(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)
i_max, p_max, S_max, U_max, C_max = sol_max.y

print(f'Cost for optimised solution: {J(opt_switching_times)}')
print(f'Cost for non-optimal solution: {J([0])}')

plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(t_span, S_opt, label='S (Optimised)', color='green')
ax1.plot(t_span, S_max, label='S (Max Control)', color='green', linestyle='dashed')
ax1.axhline(S_min, color='red', linestyle='dashed', label='S (minimum target)')
ax1.set_ylim(0, 100000)
ax1.set_ylabel('S (Money)')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(t_span, i_opt, label='i (Optimised)', color='blue')
ax2.plot(t_span, p_opt, label='p (Optimised)', color='red')
ax2.plot(t_span, i_max, label='i (Max Control)', color='blue', linestyle='dashed')
ax2.plot(t_span, p_max, label='p (Max Control)', color='red', linestyle='dashed')
ax2.set_ylabel('i, p')
ax2.legend(loc='upper right')
plt.title('S, i, p Over Time')

plt.subplot(1, 2, 2)
plt.plot(t_span, [u_opt(ti) for ti in t_span], label='u_opt', color='purple')
plt.plot(t_span, [u_always_max(ti) for ti in t_span], label='Max control', color='purple', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Control')
plt.title('Control u Over Time')
plt.legend()

plt.tight_layout()
plt.show()
