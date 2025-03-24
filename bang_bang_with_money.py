import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from simulation.finance_data import FinanceData
from simulation.parameters_calculator import ParameterCalculator
from scipy.optimize import differential_evolution


data = FinanceData()
data.download()

interest_calculator = ParameterCalculator(rp= lambda t: 0.3, rr= lambda _: 0.0)#lambda t: 2*data.interpolated_r_r(t))#lambda t: data.interpolated_r_r(t))#data.interpolated_r_r)
interest_calculator.compute_sentiment(0, 30, 100, np.log(4) / 2)
num_switches = 6
# Example, adjust as needed

#lambda_ = lambda _: 0.03#interest_calculator.lambda_from_rr_func(base=0.2, min=0.05, max=0.2, steepness=30)


#rp = interest_calculator.r_p
def mu(t):
    #return np.where(18 <= t <= 20, 0.6, np.where(28<=t<=30, 1, 0.05))
    #return 0.05
    return 0.1#interest_calculator.mu_from_rr_func(base=0.1, min=0.03, max=0.9, steepness=150)(t)

u_min, u_max = 0, 0.02
def rp(u):
    return 0.25#u_max#0.25
    #return u

def lambda_(u, t):
    return u/u_max * 0.030#interest_calculator.lambda_from_rr_func(max=0.1, steepness=10)(t)
    #rp_ = rp(u)
    #return 0.05 + rp_/u_max * 0.05#rp / u_max * 0.04
    #np.where(0.03 * rp(t), )

M = 100  # Define constants if necessary
N = 10000
avg_k = 10




def system_dynamics(t, y, u):

    i, p, S, U, C = y
    r_p = rp(u)
    lambda_eval = lambda_(u, t)
    #print(f'at {t} lambda is {lambda_eval}, mu is {mu(t)}')
    #print(f'pars: {i}, {p}, {S}, {U}, {C}, {N}, {M}, {avg_k}, {interest_calculator.r_p(t)}, {interest_calculator.r_r(t)}')
    mu_eval = mu(t)
    avg_w = U / C if C != 0 else 0

    di_dt = i * p * avg_k * lambda_eval - mu_eval * i
    dp_dt = -i * p * avg_k * lambda_eval
    dS_dt = (S * interest_calculator.rr(t) + M * N * lambda_eval * p * avg_k * i) - N * mu_eval * i * avg_w
    dU_dt = lambda_eval * avg_k * p * i * M + (r_p - mu_eval) * U
    dC_dt = lambda_eval * avg_k * p * i - mu_eval * C

    return [di_dt, dp_dt, dS_dt, dU_dt, dC_dt]


def optimal_control(params):
    """Generates the control function based on optimized switching times and values."""
    num_switches = len(params) // 2
    switching_times = np.sort(params[:num_switches])  # Ensure times are sorted
    switching_values = np.clip(params[num_switches:], u_min, u_max)  # Ensure values are within [u_min, u_max]

    def control_function(t):
        """Returns u(t) based on the switching schedule."""
        t = np.array(t)
        control_values = np.zeros_like(t)

        for i in range(num_switches - 1):
            mask = (t >= switching_times[i]) & (t < switching_times[i + 1])
            control_values[mask] = switching_values[i]

        control_values[t >= switching_times[-1]] = switching_values[-1]  # Last value

        return control_values

    return control_function


i0, p0, S0, U0, C0 = 1./N, 1 - 1./N, 5000, 0, 0
t_span = np.linspace(0, 30, 100)
#print([interest_calculator.r_r(ti) for ti in t_span])

S_min = 100000
def lagrangian(y):
    i,p,S,U,C = y
    return -S#np.where((S > S_min), -S, 0)#-S
    #return np.where((S > S_min), 0, 1)
    #return np.where((S > S_min) & (i > i_threshold), 0, 1) - S

n_iter =0
def J(switching_times):
    global n_iter
    u = optimal_control(switching_times)
    sol = solve_ivp(lambda t, y: system_dynamics(t, y, u(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)

    i, p, S, U, C = sol.y
    #cost = np.sum(lagrangian(sol.y)) * (t_span[1] - t_span[0])

    start_idx, stop_idx = find_range(S)
    #print('computed start, stop ', start_idx, stop_idx)
    valid_range = sol.y[:, :]

    # Compute the cost only in the valid range
    #print('considered array is ', )
    if (len(valid_range) == 0):
        return 0
    cost = S[-1] + np.sum(lagrangian(valid_range) * (t_span[1] - t_span[0]))
    #print('computed cost is ', cost)
    n_iter += 1
    return cost


def find_range(variable):
    cross_below_idx = np.where((variable[:-1] > S_min) & (variable[1:] <= S_min))[0]
    cross_above_idx = np.where((variable[:-1] <= S_min) & (variable[1:] > S_min))[0]

    if cross_above_idx.size > 0:
        start_idx = cross_above_idx[0]  # First time S crosses above the threshold
    else:
        return 0, 0  # If S never crosses above, return (0, 0)

    if cross_below_idx.size > 0:
        stop_idx = cross_below_idx[0] + 1  # First time S crosses below the threshold
    else:
        stop_idx = len(variable)  # If S never goes below, take the full range

    return start_idx, stop_idx


# initial_guess = np.zeros(5)#np.linspace(0, 30, 2)
# sol = minimize(lambda switching_times: J(switching_times),
#                initial_guess,
#                bounds=[(0, 30)] * len(initial_guess),
#                tol=1e-10,
#                options={'maxiter': 500, 'xtol': 1e-2},
#                method='Powell')
bounds = [(0, 30)] * num_switches + [(u_min, u_max)] * num_switches
sol = differential_evolution(J, bounds, strategy='best1bin', tol=1e-3, maxiter=2)
opt_switching_times = sol.x
#
# lb = [0] * len(initial_guess)   # Lower bounds
# ub = [30] * len(initial_guess)  # Upper bounds
#
# # Run PSO
# opt_switching_times, best_cost = pso(J, lb, ub, swarmsize=20, maxiter=200)

u_opt = optimal_control(opt_switching_times)
sol_opt = solve_ivp(lambda t, y: system_dynamics(t, y, u_opt(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)
i_opt, p_opt, S_opt, U_opt, C_opt = sol_opt.y

u_always_max = optimal_control([0, u_max])
sol_max = solve_ivp(lambda t, y: system_dynamics(t, y, u_always_max(t)), (0, 30), [i0, p0, S0, U0, C0], t_eval=t_span)
i_max, p_max, S_max, U_max, C_max = sol_max.y

print(f'Cost for optimised solution: {J(opt_switching_times)}')
print(f'Cost for non-optimal solution: {J([0, u_max])}')

plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(t_span, S_opt, label='S (Optimised)', color='green')
ax1.plot(t_span, S_max, label='S (Max Control)', color='green', linestyle='dashed')
#ax1.axhline(S_min, color='red', linestyle='dashed', label='S (minimum target)')
#ax1.set_ylim(0, 1000000)
ax1.set_ylabel('S (Money)')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(t_span, i_opt, label='i (Optimised)', color='blue')
#ax2.plot(t_span, p_opt, label='p (Optimised)', color='red')
ax2.plot(t_span, i_max, label='i (Max Control)', color='blue', linestyle='dashed')
#ax2.plot(t_span, p_max, label='p (Max Control)', color='red', linestyle='dashed')
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
