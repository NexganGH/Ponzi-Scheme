from networks.interest_calculator import InterestCalculator
import numpy as np
from networks import Network, Node
from networks.node_status import NodeStatus


class PonziSimulation:
    def __init__(self,
                 network: Network,
                 interest_calculator: InterestCalculator,
                 max_time_units: int = 10000,
                 dt: float = 1. / 12,
                 lambda_ = lambda t: 0.1,
                 mu = lambda t: 0.1,
                 capital_per_person: float = 100, ponzi_capital: float = 100) -> None:
        self.network = network
        self.interest_calculator = interest_calculator
        self.max_time_units = max_time_units
        self.dt = dt
        self.lambda_ = lambda_
        self.mu = mu
        self.capital_per_person = capital_per_person
        self.ponzi_capital = ponzi_capital

    def _update_counts(self, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time):
        # Efficiently calculate counts using numpy
        nodes = self.network.nodes

        statuses = np.array([node.status for node in nodes])
        degrees = np.array([node.degree for node in nodes])

        investor_numbers.append(np.sum(statuses == NodeStatus.INVESTOR))
        potential_numbers.append(np.sum(statuses == NodeStatus.POTENTIAL))
        deinvestor_numbers.append(np.sum(statuses == NodeStatus.DEINVESTOR))

        for d in range(degrees_money.shape[0]):
            mask = (degrees == d) #(statuses == NodeStatus.INVESTOR) & (degrees == d)
            if np.any(mask):
                degrees_money[d, time] = self.network.capital_array[mask].sum() / np.sum(mask)

    def simulate_ponzi(self):
        print(f'Starting simulation with lambda={self.lambda_}, mu={self.mu}')
        time = 0
        ponzi = self.network.ponzi_node()
        nodes = self.network.nodes
        ponzi.capital = self.ponzi_capital


        # Logs
        ponzi_capital = [ponzi.capital]
        investor_numbers, potential_numbers, deinvestor_numbers = [], [], []
        degrees_money = np.zeros((50, self.max_time_units))

        self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)

        last_signal, signal_every = 0, 0.05
        while time < self.max_time_units and ponzi.capital / self.ponzi_capital >= -10:
            perc = time / self.max_time_units
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            if time > 0:
                ponzi.capital = self.interest_calculator.realized_return(ponzi.capital,
                                                                    (time - 1) * self.dt,
                                                                    time * self.dt)
            for i in range(1, len(nodes)):  # Skip the Ponzi node
                self.evolve_node(i, nodes[i], time)

            # Update logs
            ponzi_capital.append(ponzi.capital)
            self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)
            time += 1

        return [ponzi_capital, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money]

    def evolve_node(self, i: int, node: Node, time: float):
            ponzi = self.network.ponzi_node()

            if node.status == NodeStatus.INVESTOR:
                # if time % self.interest_calculating_periods == 0:
                #   interest = self._money_per_turn()
                #   self.capital_array[i] += interest
                #  ponzi.capital -= interest

                if np.random.binomial(1, self.mu(time)):  # Node exits
                    exit_capital = self.interest_calculator.promised_return_at_time(self.capital_per_person,
                                                                               node.time_joined * self.dt,
                                                                               time * self.dt)  # self.capital_per_person
                    print(f'joined at {node.time_joined * self.dt}, left at {time*self.dt} with {exit_capital}')
                    self.network.capital_array[i] += exit_capital
                    ponzi.capital -= exit_capital
                    node.status = NodeStatus.DEINVESTOR
            elif node.status == NodeStatus.POTENTIAL:
                for connection in node.connections:
                    if connection.status == NodeStatus.INVESTOR and np.random.binomial(1, self.lambda_(time)):
                        invest_capital = self.capital_per_person
                        self.network.capital_array[i] -= invest_capital
                        ponzi.capital += invest_capital
                        node.make_investor(connection, time)
