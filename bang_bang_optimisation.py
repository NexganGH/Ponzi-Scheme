import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from simulation.finance_data import FinanceData
from simulation.parameters_calculator import InterestCalculator

data = FinanceData()
data.download()

interest_calculator = InterestCalculator(rp=lambda _: 0.1, rr=data.interpolated_r_r)
interest_calculator.compute_market_positivity(0, 30, 200)



lambda_ = interest_calculator.lambda_from_rr_func(base=0.1, min=0, max=0.1, steepness=30)
#mu = 0.4
def mu(t):
    return interest_calculator.mu_from_rr_func(0.2, 0.1, 0.6, 150)(t)
    return np.where(t < 6, 0.4, np.where(t<10, 0.8, 0.4))
k = 5

u_min = 0
u_max = 0.1


def system_dynamics(t, y, u):
    # Extract the state variables (modify according to your equations)
    i, p = y

    #u = u_max if lambda1 > 0 else u_min

    # Define the derivatives (modify according to your system)
    lambda_eval = lambda_(t)
    di_dt = i * p * k * (lambda_eval + u) - mu(t) * i   # Example equation for i
    dp_dt = -i * p * k * (lambda_eval + u)  # Example equation for p

    return [di_dt, dp_dt]

def optimal_control(switching_times):
    def control_function(t):
        """
        Determines the control value based on time t using a bang-bang control strategy.
        """
        # Ensure switching_times is sorted
        switching_times_sorted = sorted(switching_times)
        t = np.array(t)
        # Initialize control array for the time points
        control_values = np.zeros_like(t)

        for i in range(len(switching_times_sorted) - 1):
            # Switch between u_min and u_max based on the index of the interval
            if i % 2 == 1:
                # Odd indexed intervals (0, 2, 4, ...) use u_min
                control_values[(t >= switching_times_sorted[i]) & (t < switching_times_sorted[i + 1])] = u_min
            else:
                # Even indexed intervals (1, 3, 5, ...) use u_max
                control_values[(t >= switching_times_sorted[i]) & (t < switching_times_sorted[i + 1])] = u_max

        # Handle the case for the last interval (if there's an odd number of switching times)
        if len(switching_times_sorted) % 2 == 1:
            control_values[t >= switching_times_sorted[-1]] = u_max

        return control_values
    return control_function

i_min = 0.1
def lagrangian(t, y):
    i, p = y
    return np.where(i <= i_min, 1, 0)

i0 = 0.01
p0 = 1 - i0
t_span = np.linspace(0, 30, 100)
# We compute the cost, which will need to be minimised
def J(switching_times):
    u = optimal_control(switching_times)
    sol = solve_ivp(lambda t, y: system_dynamics(t, y, u(t)), (0, 30), [i0, p0], t_eval=t_span)
    i, p = sol.y
    #return np.sum(lagrangian(t, sol.y))
    #return quad(lambda t: lagrangian(t, sol.y), 0, 30)
    cost = np.sum(lagrangian(t_span, sol.y)) * (t_span[1] - t_span[0])
    #print(lagrangian(t_span, sol.y), t_span)
    print('found cost ', cost, ' with pars ', switching_times)
    return cost

# best_switching_times, best_cost = None, np.inf
# for initial_guess in np.linspace(0, 30, 20):
#     print('testing ', initial_guess)
#     initial_guess = [initial_guess]
#     sol = minimize(lambda switching_times: J(switching_times), initial_guess, bounds=[(0, 30)] * len(initial_guess), tol=1e-6, options={'maxiter':100, 'gtol':1e-1}, method='BFGS')
#     cost = J(sol.x)
#
#     if cost < best_cost:
#         best_cost = cost
#         best_switching_times = sol.x
#
# opt_switching_times = best_switching_times
# print('Best switching times for k are are ', opt_switching_times, ' with cost ', best_cost)

initial_guess = np.zeros(5)
sol = minimize(lambda switching_times: J(switching_times),
               initial_guess,
               bounds=[(0, 30)] * len(initial_guess),
               tol=1e-6,
               options={'maxiter': 500, 'xtol': 1e-1},
               method='Powell')

opt_switching_times = sol.x
u_opt = optimal_control(opt_switching_times)

sol_opt = solve_ivp(lambda t, y: system_dynamics(t, y, u_opt(t)), (0, 30), [i0, p0], t_eval=t_span, )
t_span = sol_opt.t
i_opt, p_opt = sol_opt.y


# ----- COMPUTING FUNCTION WITHOUT OPTIMISATION
u_always_max = optimal_control([0])
sol = solve_ivp(lambda t, y: system_dynamics(t, y, u_always_max(t)), (0, 30), [i0, p0], t_eval=t_span, )
i_max, p_max = sol.y


# ----- COMPUTING COSTS

print(f'Cost for optimised solution: {J(opt_switching_times)}')
print(f'Cost for non-optimal solution: {J([0])}')


# --------------- PLOTS

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_span, i_opt, label='i_opt', color='blue')
plt.plot(t_span, p_opt, label='p_opt', color='red')
plt.axhline(i_min, color='black', label='target i')
plt.xlabel('Time')
plt.ylabel('States')
plt.title('i_opt and p_opt')


plt.plot(t_span, i_max, label='i Max control', color='blue', linestyle='dashed')
plt.plot(t_span, p_max, label='i max control', color='red', linestyle='dashed')

plt.legend()

# Plot u_opt in the second graph
plt.subplot(1, 2, 2)
plt.plot(t_span, [u_opt(ti) for ti in t_span], label='u_opt', color='green')

plt.plot(t_span, [u_always_max(ti) for ti in t_span], label='Max control', color='green', linestyle='dashed')

plt.xlabel('Time')
plt.ylabel('Control')
plt.title('u_opt')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()



# Now we compare the costs
