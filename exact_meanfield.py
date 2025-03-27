import numpy as np
import scipy.integrate as spi
import scipy.interpolate as spi_interp
import scipy.stats as stats
from simulation import SimulationResult, PonziPlotter


def calculate_i_p_distribution(P, k_min, k_max):
    #global m, lambda_, mu, avg_k, theta, i_integral_k, ponzi_results, k, p_k_results
    m = 5.5 # Example value for m
    lambda_ = 0.2  # Infection rate
    mu = 0.1  # Recovery rate
    avg_k = 6

    def theta(z):
        sum_over_k = 0
        for k in range(k_min, k_max):
            sum_over_k += k / avg_k * P(k) * np.exp(-k * z / m)
        return 1 - 1 * sum_over_k - mu / (lambda_ * m) * z

    # Define function F(z) based on the given parameters
    def F(z):
        return lambda_ * m * theta(z)

    # Define function for potential investors p_k
    def p_k(k, z):
        return np.exp(-z * k / m)

    def i_integral_k(k, tau, z_func):
        return np.exp(mu * tau) * k * lambda_ * np.exp(-k * z_func(tau) / m) * theta(z_func(tau))

    def i_k(k, z_func, t_values):
        val = spi.cumulative_trapezoid(i_integral_k(k, t_values, z_func), t_values, initial=0)
        # print(i_k(k, val, t_values))
        return np.exp(-mu * t_values) * val

    z_values = np.linspace(0., 9, 100000)  # Range of z values
    t_values = spi.cumulative_trapezoid(1 / F(z_values), z_values, initial=0)  # Ensure t(0) = 0
    z_of_t = spi_interp.interp1d(t_values, z_values, kind='cubic', fill_value='extrapolate')
    ponzi_results = {}
    # Compute p_k over time for each k
    t_plot = np.linspace(0, 30, 100000)
    p_k_results = {k: p_k(k, z_of_t(t_plot)) for k in range(k_min, k_max)}
    # Compute total p over time
    p_total_plot = 0
    for (k, p_k) in p_k_results.items():  # Summing over k from 2 to 100
        p_total_plot += P(k) * p_k
    # Compute i_k over time for each k
    i_k_results = {k: i_k(k, z_of_t, t_plot) for k in range(k_min, k_max)}
    i_total_plot = 0
    for (k, i_k) in i_k_results.items():
        i_total_plot += P(k) * i_k
    # Create PonziResults for each k
    for k in p_k_results:
        ponzi_results[k] = SimulationResult(
            investor_numbers=np.array(i_k_results[k]),  # p_k represents investors for each k
            potential_numbers=np.array(p_k_results[k]),  # Placeholder if needed
            deinvestor_numbers=[0] * len(i_k_results[k]),
            capital=[0] * len(i_k_results[k]),
            # deinvestor_numbers=np.zeros_like(p_k_results[k]),  # Placeholder if needed
            # capital=np.array(p_total_plot),  # Use total p as capital proxy
            dt=t_plot[1] - t_plot[0]  # Time step
        )
    # Aggregate all data into k=0 (all nodes)
    ponzi_results[0] = SimulationResult(
        investor_numbers=np.array(i_total_plot),  # Total p represents global investors
        potential_numbers=np.array(p_total_plot),  # Placeholder
        deinvestor_numbers=[0] * len(p_total_plot),  # np.zeros_like(p_total_plot),  # Placeholder
        capital=[0] * len(i_total_plot),
        dt=t_plot[1] - t_plot[0]
    )
    return ponzi_results

def P_er(k):
    return stats.poisson.pmf(k, 6)

def P_ba(k):
    # return stats.poisson.pmf(k, avg_k)
    gamma = 2.47
    k_min = 2
    k_max = 100
    norm_constant = sum(k ** (-gamma) for k in range(k_min, k_max + 1))

    # Calculate the probability for each degree k
    return (k ** (-gamma)) / norm_constant if k >= k_min else 0
er_results = calculate_i_p_distribution(P_er, 0, 20)
er_plot = PonziPlotter()
for k in [0, 3, 6, 10]:
    res = er_results[k]
    lab = 'Int. - k={}'.format(k) if k != 0 else "Int. - Totale"
    er_plot.add_simulation(res, label=lab)

er_plot.plot(show_investor=True, show_potential=True)
er_plot.save_data('data/er_mean_field.csv')
# Define parameters
ba_results = calculate_i_p_distribution(P_ba, 2, 101)
ba_plot = PonziPlotter()
for k in [0, 6, 10, 50]:
    res = ba_results[k]
    lab = 'Int. - k={}'.format(k) if k != 0 else "Int. - Totale"
    ba_plot.add_simulation(res, label=lab)
ba_plot.plot(show_investor=True, show_potential=True)
ba_plot.save_data('data/ba_mean_field.csv')
