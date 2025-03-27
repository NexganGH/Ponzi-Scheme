import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Callable
import pandas as pd
from .simulation_result import SimulationResult
font_size = 15

class PonziPlotter:
    def __init__(self):
        self.simulations = []  # Store all simulation results
        self.markers = ['o', '*', '>', '^', 'v', '<', 'D', 'p', 's', 'h']  # Different markers

    def add_simulation(self, simulation_result, label: str = None, custom_color: str = None):
        """Adds a simulation result to the plotter."""
        self.simulations.append((simulation_result, label, custom_color))
        return self

    def plot(self, file_name='asd', title='Sistema', show_potential=False,
             show_investor=False, show_deinvestor=False, show_capital=False,
             max_capital=10000, custom_func: Callable[[plt.axes], None] = None, offset=0):
        """Plots all added simulation results on the same graph."""
        if not self.simulations:
            raise ValueError("No simulations to plot. Add simulations first.")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Lists to hold handles and labels for the legends
        lines_investitori = []
        lines_potentiali = []
        lines_deinvestitori = []
        lines_capitale = []
        markers = []

        legend_marker_lines = []
        legend_marker_labels = []
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
                line_investitori, = ax1.plot(t + offset, i, color=color, marker=marker, linestyle='-', markersize=8,
                                             markevery=markevery)
                lines_investitori.append(line_investitori)

            if show_potential:
                line_potentiali, = ax1.plot(t + offset, p, color='green', marker=marker, linestyle='-', markersize=8,
                                            markevery=markevery)
                lines_potentiali.append(line_potentiali)

            if show_deinvestor:
                line_deinvestitori, = ax1.plot(t + offset, d, color='gray', marker=marker, linestyle='-', markersize=8,
                                               markevery=markevery)
                lines_deinvestitori.append(line_deinvestitori)

            # Secondary y-axis plots (capital evolution) with dashed line
            if show_capital:
                max_cap = max(max_cap, max(capital))
                color = 'red' if not custom_color else custom_color
                line_money, = ax1.plot(t + offset, capital, linestyle='dashed', color=color, alpha=0.7, marker=marker, markersize=8,
                                       markevery=markevery)
                lines_capitale.append(line_money)

            legend_marker_lines.append(Line2D([0], [0], color='black', lw=0, marker=marker, markersize=8))
            legend_marker_labels.append(label)

        # Set labels for each axis
        ax1.set_xlabel('t', fontsize=font_size)
        ax1.set_ylabel('Popolazione' if not show_capital else 'Capitale S(t)', fontsize=font_size)
        ax1.set_ylim(bottom=0, top=max_cap if show_capital else 1)
        legend_lines = []
        legend_labels = []
        ax1.tick_params(axis='x', which='both', labelsize=font_size)  # Per l'asse X
        ax1.tick_params(axis='y', which='both', labelsize=font_size)  # Per l'asse Y

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

        leg1 = ax1.legend(legend_lines, legend_labels, loc='upper left', fontsize=font_size)
        leg2 = ax1.legend(legend_marker_lines, legend_marker_labels, loc='upper right', fontsize=font_size)
        ax1.add_artist(leg1)
        ax1.add_artist(leg2)

        ax1.grid()
        if custom_func is not None:
            custom_func(ax1)
        plt.title(f'{title}', fontsize=font_size + 2)
        plt.savefig(f'imgs/{file_name}.png', bbox_inches='tight')
        plt.show()

    def save_data(self, file_name: str):
        """Saves the simulation data to a CSV file."""
        if not self.simulations:
            raise ValueError("No simulations to save. Add simulations first.")

        # Prepare a DataFrame to store the data
        data = {
            'time': [],
            'investors': [],
            'potential': [],
            'deinvestors': [],
            'capital': [],
            'label': []
        }

        # Loop through each simulation and add the data
        for simulation_result, label, _ in self.simulations:
            i, p, d, capital = (simulation_result.investor_numbers,
                                simulation_result.potential_numbers,
                                simulation_result.deinvestor_numbers,
                                simulation_result.capital)
            t = np.arange(len(capital)) * simulation_result.dt

            # Add the data to the dictionary
            data['time'].extend(t)
            data['investors'].extend(i)
            data['potential'].extend(p)
            data['deinvestors'].extend(d)
            data['capital'].extend(capital)
            data['label'].extend([label] * len(t))  # Repeat the label for each time step

        # Create a DataFrame and save it to a CSV file
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")

    def load_data(self, file_name: str):
        """Loads simulation data from a CSV file and adds it to the plotter."""
        df = pd.read_csv(file_name)

        # Check that necessary columns exist in the file
        required_columns = ['time', 'investors', 'potential', 'deinvestors', 'capital', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file must contain columns: 'time', 'investors', 'potential', 'deinvestors', 'capital', and 'label'.")

        # Group by label and reconstruct simulation data
        for label, group in df.groupby('label'):
            time = group['time'].values
            investors = group['investors'].values
            potential = group['potential'].values
            deinvestors = group['deinvestors'].values
            capital = group['capital'].values

            # Assuming simulation_result is a named tuple or class with these attributes
            simulation_result = SimulationResult(investors, potential, deinvestors, capital, dt=(time[1] - time[0]))  # Estimate dt
            self.add_simulation(simulation_result, label=label)

        print(f"Data loaded from {file_name}")
