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

    def simulate_ponzi(self, computed_for_each_k=False):
        """"Avvia la simulazione"""
        print(f'Starting simulation')
        time = 0
        ponzi = self.network.ponzi_node()
        nodes = self.network.nodes
        ponzi.capital = self.parameters.starting_capital

        ponzi_capital_numbers = []
        investor_per_k = {0: []}
        potential_per_k = {0: []}
        deinvestor_per_k = {0: []}

        # Aggiungi tutti i gradi del network
        all_degrees = {node.k() for node in nodes[1:]}
        for k in all_degrees:
            investor_per_k[k] = []
            potential_per_k[k] = []
            deinvestor_per_k[k] = []

        last_signal, signal_every = 0, 0.05
        while time < self.max_time_units:
            perc = time / self.max_time_units
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            # STEP 1: Calcola l'interesse accumulato
            if time > 0:
                new_capital = self.interest_calculator.ponzi_earnings(
                    ponzi.capital, (time - 1) * self.dt, time * self.dt
                )
                ponzi.capital = new_capital

            # Salva gli investiori correnti per grado
            current_investors_per_k = {k: 0 for k in investor_per_k}
            current_potentials_per_k = {k: 0 for k in investor_per_k}
            current_deinvestors_per_k = {k: 0 for k in investor_per_k}

            # STEP 2: Evolve e salva i nodi.
            for i in range(1, len(nodes)):

                self.evolve_node(i, nodes[i], time)
                k = nodes[i].k()
                if nodes[i].status == NodeStatus.INVESTOR:
                    current_investors_per_k[k] += 1
                    current_investors_per_k[0] += 1
                elif nodes[i].status == NodeStatus.POTENTIAL:
                    current_potentials_per_k[k] += 1
                    current_potentials_per_k[0] += 1
                elif nodes[i].status == NodeStatus.DEINVESTOR:
                    current_deinvestors_per_k[k] += 1
                    current_deinvestors_per_k[0] += 1

            ponzi_capital_numbers.append(ponzi.capital)

            for k in investor_per_k:
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
        results = {}
        for k in investor_per_k:
            results[k] = SimulationResult(
                investor_numbers=np.array(investor_per_k[k]),
                potential_numbers=np.array(potential_per_k[k]),
                deinvestor_numbers=np.array(deinvestor_per_k[k]),
                capital=ponzi_capital_numbers,
                dt=self.dt
            )

        if not computed_for_each_k:
            return results[0]  # Dictionary of SimulationResult per k, with k=0 representing all nodes
        return results

    def evolve_node(self, i: int, node: Node, time: float):
            ponzi = self.network.ponzi_node()

            # Se è investitore, può uscire dallo schema con una prob. mu * Delta t
            if node.status == NodeStatus.INVESTOR:
                if np.random.binomial(1, self.mu(time*self.dt)*self.dt):  # Node exits
                    exit_capital = self.interest_calculator.promised_return_at_time(self.M,
                                                                                    node.time_joined * self.dt,
                                                                               time * self.dt)  # self.capital_per_person
                    self.network.capital_array[i] += exit_capital
                    ponzi.capital -= exit_capital
                    node.status = NodeStatus.DEINVESTOR
            # Se è potenziale investitore, iteriamo sulle possibili connessioni:
            # per ognuna diventa investitore con una prob lambda * Delta t
            elif node.status == NodeStatus.POTENTIAL:
                for connection in node.connections:
                    if connection.status == NodeStatus.INVESTOR and np.random.binomial(1, self.lambda_(time*self.dt)*self.dt):
                        invest_capital = self.M
                        self.network.capital_array[i] -= invest_capital
                        ponzi.capital += invest_capital
                        node.make_investor(connection, time)
                        break # Altrimenti potrebbe diventare investitore due volte.