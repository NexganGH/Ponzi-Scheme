import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class PonziPlotter:
    def __init__(self):
        self.simulations = []  # Store all simulation results
        self.markers = ['o', '*', '>', '^', 'v', '<', 'D', 'p', 's', 'h']  # Different markers

    def add_simulation(self, simulation_result, label: str = None,
                       ):
        """Adds a simulation result to the plotter."""
        self.simulations.append((simulation_result, label))

    def plot(self, file_name='asd', title='Sistema', show_potential=False,
                       show_investor=False,
                       show_deinvestor=False,
                       show_capital=False):
        """Plots all added simulation results on the same graph."""
        if not self.simulations:
            raise ValueError("No simulations to plot. Add simulations first.")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Lists to hold handles and labels for the legends
        lines_investitori = []
        lines_potentiali = []
        lines_deinvestitori = []
        lines_money = []
        markers = []


        legend_marker_lines = []
        legend_marker_labels=[]
        for idx, (simulation_result, label) in enumerate(self.simulations):
            i, p, d, capital = (simulation_result.investor_numbers,
                                simulation_result.potential_numbers,
                                simulation_result.deinvestor_numbers,
                                simulation_result.capital)
            t = np.arange(len(capital)) * simulation_result.dt
            marker = self.markers[idx % len(self.markers)]  # Cycle through markers

            # Reduce marker density by only plotting every 5th point, for example
            markevery = len(t) // 10  # Adjust this to change marker frequency

            # Primary y-axis plots (population dynamics) with consistent line style
            if show_investor:
                line_investitori, = ax1.plot(t, i, color='blue', marker=marker, linestyle='-', markersize=8,
                                             markevery=markevery)
                lines_investitori.append(line_investitori)

            if show_potential:
                line_potentiali, = ax1.plot(t, p, color='green', marker=marker, linestyle='-', markersize=8,
                                            markevery=markevery)
                lines_potentiali.append(line_potentiali)

            if show_deinvestor:
                line_deinvestitori, = ax1.plot(t, d, color='gray', marker=marker, linestyle='-', markersize=8,
                                               markevery=markevery)
                lines_deinvestitori.append(line_deinvestitori)

            # Secondary y-axis plots (capital evolution) with dashed line
            if show_capital:
                line_money, = ax2.plot(t, capital, linestyle='dashed', color='red', alpha=0.7, marker=marker, markersize=8,
                                       markevery=markevery)
                lines_money.append(line_money)

            legend_marker_lines.append(Line2D([0], [0], color='black', lw=0, marker=marker, markersize=8))
            legend_marker_labels.append(label)
        # Set labels for each axis
        ax1.set_xlabel('t')
        ax1.set_ylabel('Popolazione')
        ax2.set_ylabel('Money')
        ax2.legend(legend_marker_lines, legend_marker_labels, loc='upper right')
        ax2.set_ylim(bottom=0)
        legend_lines = []
        legend_labels = []

        # Create the legends
        if len(lines_investitori) > 0:
            legend_lines.append(Line2D([0], [0], color='blue', lw=2, label=f'Investitori'))
            legend_labels.append('Investitori')
        if len(lines_potentiali) > 0:
            legend_lines.append(Line2D([0], [0], color='green', lw=2, label=f'Potenziali'))
            legend_labels.append('Potenziali')
        if len(lines_deinvestitori) > 0:
            legend_lines.append(Line2D([0], [0], color='gray', lw=2, label=f'Deinvestitori'))
            legend_labels.append('Deinvestitori')

        ax1.legend(legend_lines, legend_labels, loc='upper left')
        ax1.grid()
        plt.title('Sistema')
        plt.savefig(f'imgs/{file_name}.png', bbox_inches='tight')
        plt.show()
