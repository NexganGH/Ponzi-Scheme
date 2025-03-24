import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Callable


class PonziPlotter:
    def __init__(self):
        self.simulations = []  # Store all simulation results
        self.markers = ['o', '*', '>', '^', 'v', '<', 'D', 'p', 's', 'h']  # Different markers

    def add_simulation(self, simulation_result, label: str = None, custom_color: str = None):
        """Adds a simulation result to the plotter."""
        self.simulations.append((simulation_result, label, custom_color))

    def plot(self, file_name='asd', title='Sistema', show_potential=False,
                       show_investor=False,
                       show_deinvestor=False,
                       show_capital=False,
                        max_capital=10000,
                        custom_func: Callable[[plt.axes], None] = None, offset=0):
        """Plots all added simulation results on the same graph."""
        if not self.simulations:
            raise ValueError("No simulations to plot. Add simulations first.")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        #ax2 = ax1.twinx()

        # Lists to hold handles and labels for the legends
        lines_investitori = []
        lines_potentiali = []
        lines_deinvestitori = []
        lines_capitale = []
        markers = []


        legend_marker_lines = []
        legend_marker_labels=[]
        max_cap = max_capital
        for idx, (simulation_result, label, custom_color) in enumerate(self.simulations):
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
                color = 'blue' if not custom_color else custom_color
                line_investitori, = ax1.plot(t+offset, i, color=color, marker=marker, linestyle='-', markersize=8,
                                             markevery=markevery)
                lines_investitori.append(line_investitori)

            if show_potential:
                line_potentiali, = ax1.plot(t+offset, p, color='green', marker=marker, linestyle='-', markersize=8,
                                            markevery=markevery)
                lines_potentiali.append(line_potentiali)

            if show_deinvestor:
                line_deinvestitori, = ax1.plot(t+offset, d, color='gray', marker=marker, linestyle='-', markersize=8,
                                               markevery=markevery)
                lines_deinvestitori.append(line_deinvestitori)

            # Secondary y-axis plots (capital evolution) with dashed line
            if show_capital:
                max_cap = max(max_cap, max(capital))
                color = 'red' if not custom_color else custom_color
                line_money, = ax1.plot(t+offset, capital, linestyle='dashed', color=color, alpha=0.7, marker=marker, markersize=8,
                                       markevery=markevery)
                lines_capitale.append(line_money)

            legend_marker_lines.append(Line2D([0], [0], color='black', lw=0, marker=marker, markersize=8))
            legend_marker_labels.append(label)
        # Set labels for each axis
        ax1.set_xlabel('t')
        ax1.set_ylabel('Popolazione' if not show_capital else 'Capitale S(t)')
        ax1.set_ylim(bottom=0, top=max_cap if show_capital else 1)
        # ax2.set_ylabel('Capitale')
        # ax2.set_ylim(bottom=0)
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
        if len(lines_capitale) > 0:
            legend_lines.append(Line2D([0], [0], color='red', lw=2, label=f'Capitale'))
            legend_labels.append('Capitale')

        leg1 = ax1.legend(legend_lines, legend_labels, loc='upper left')
        leg2 = ax1.legend(legend_marker_lines, legend_marker_labels, loc='upper right')
        ax1.add_artist(leg1)
        ax1.add_artist(leg2)


        ax1.grid()
        if custom_func is not None:
            custom_func(ax1)
        plt.title(f'{title}')
        plt.savefig(f'imgs/{file_name}.png', bbox_inches='tight')
        plt.show()
