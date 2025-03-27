import numpy as np
from networks import Network, Node
from networks.node_status import NodeStatus
from .simulation_result import SimulationResult
from .ponzi_parameters import PonziParameters


class PonziSimulation:
    def __init__(self,
                 network: Network,
                 parameters: PonziParameters,
                 max_time_units: int = 10000,
                 dt: float = 1. / 12) -> None:
        self.network = network
        self.max_time_units = max_time_units
        self.dt = dt
        self.parameters = parameters

        self.lambda_ = parameters.lambda_
        self.mu = parameters.mu
        self.ponzi_capital = parameters.starting_capital
        self.interest_calculator = parameters.interest_calculator
        self.M = parameters.M

    def _update_counts(self, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time):
        # Efficiently calculate counts using numpy
        nodes = self.network.nodes

        statuses = np.array([node.status for node in nodes])
        degrees = np.array([node.k() for node in nodes])

        investor_numbers.append(np.sum(statuses == NodeStatus.INVESTOR))
        potential_numbers.append(np.sum(statuses == NodeStatus.POTENTIAL))
        deinvestor_numbers.append(np.sum(statuses == NodeStatus.DEINVESTOR))

        for d in range(degrees_money.shape[0]):
            mask = (degrees == d) #(statuses == NodeStatus.INVESTOR) & (degrees == d)
            if np.any(mask):
                degrees_money[d, time] = self.network.capital_array[mask].sum() / np.sum(mask)

    def simulate_ponzi(self, computed_for_each_k=False):
        print(f'Starting simulation')
        time = 0
        ponzi = self.network.ponzi_node()
        nodes = self.network.nodes
        ponzi.capital = self.parameters.starting_capital

        # Logs for global results (all nodes)
        ponzi_capital_numbers = []
        investor_numbers, potential_numbers, deinvestor_numbers = [], [], []
        degrees_money = np.zeros((50, self.max_time_units))

        if computed_for_each_k:
            # Dictionary to store time-series data for each degree, including k=0 (all nodes)
            investor_per_k = {0: []}  # k=0 represents all nodes
            potential_per_k = {0: []}
            deinvestor_per_k = {0: []}

            # Track all unique degrees appearing in the network
            all_degrees = {node.k() for node in nodes[1:]}  # Exclude Ponzi node
            for k in all_degrees:
                investor_per_k[k] = []
                potential_per_k[k] = []
                deinvestor_per_k[k] = []
            print('investors are ', investor_per_k)

        self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)

        last_signal, signal_every = 0, 0.05
        while time < self.max_time_units:
            perc = time / self.max_time_units
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            if time > 0:
                new_capital = self.interest_calculator.ponzi_earnings(
                    ponzi.capital, (time - 1) * self.dt, time * self.dt
                )
                ponzi.capital = new_capital

            # Per-degree counts if needed
            if computed_for_each_k:
                current_investors_per_k = {k: 0 for k in investor_per_k}
                current_potentials_per_k = {k: 0 for k in investor_per_k}
                current_deinvestors_per_k = {k: 0 for k in investor_per_k}

            for i in range(1, len(nodes)):  # Skip the Ponzi node
                self.evolve_node(i, nodes[i], time)
                if computed_for_each_k:
                    k = nodes[i].k()
                    if nodes[i].status == NodeStatus.INVESTOR:
                        current_investors_per_k[k] += 1
                        current_investors_per_k[0] += 1  # Aggregate all nodes
                    elif nodes[i].status == NodeStatus.POTENTIAL:
                        current_potentials_per_k[k] += 1
                        current_potentials_per_k[0] += 1
                    elif nodes[i].status == NodeStatus.DEINVESTOR:
                        current_deinvestors_per_k[k] += 1
                        current_deinvestors_per_k[0] += 1

            # Update logs for global results
            ponzi_capital_numbers.append(ponzi.capital)
            self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)

            if computed_for_each_k:
                for k in investor_per_k:
                    # Normalize by number of nodes with degree k
                    n_nodes_k = self.network.number_nodes_k(k)
                    if n_nodes_k > 0:
                        investor_per_k[k].append(current_investors_per_k[k] / n_nodes_k)
                        potential_per_k[k].append(current_potentials_per_k[k] / n_nodes_k)
                        deinvestor_per_k[k].append(current_deinvestors_per_k[k] / n_nodes_k)
                    else:
                        investor_per_k[k].append(0)
                        potential_per_k[k].append(0)
                        deinvestor_per_k[k].append(0)

            time += 1

        if computed_for_each_k:
            results = {}
            for k in investor_per_k:
                results[k] = SimulationResult(
                    investor_numbers=np.array(investor_per_k[k]),
                    potential_numbers=np.array(potential_per_k[k]),
                    deinvestor_numbers=np.array(deinvestor_per_k[k]),
                    capital=ponzi_capital_numbers,
                    dt=self.dt
                )
            return results  # Dictionary of SimulationResult per k, with k=0 representing all nodes

        return SimulationResult(
            investor_numbers=np.array(investor_numbers) / self.network.n_nodes,
            potential_numbers=np.array(potential_numbers) / self.network.n_nodes,
            deinvestor_numbers=np.array(deinvestor_numbers) / self.network.n_nodes,
            capital=ponzi_capital_numbers,
            dt=self.dt
        )

    def evolve_node(self, i: int, node: Node, time: float):
            ponzi = self.network.ponzi_node()

            if node.status == NodeStatus.INVESTOR:

                if np.random.binomial(1, self.mu(time*self.dt)*self.dt):  # Node exits
                    exit_capital = self.interest_calculator.promised_return_at_time(self.M,
                                                                                    node.time_joined * self.dt,
                                                                               time * self.dt)  # self.capital_per_person
                    #print('Entered at ', node.time_joined, ', exited at', time, ' with money ', exit_capital)
                    self.network.capital_array[i] += exit_capital
                    ponzi.capital -= exit_capital
                    node.status = NodeStatus.DEINVESTOR
            elif node.status == NodeStatus.POTENTIAL:
                for connection in node.connections:
                    if connection.status == NodeStatus.INVESTOR and np.random.binomial(1, self.lambda_(time*self.dt)*self.dt):
                        invest_capital = self.M
                        self.network.capital_array[i] -= invest_capital
                        ponzi.capital += invest_capital
                        node.make_investor(connection, time)
                        break # Altrimenti potrebbe diventare investitore due volte.


