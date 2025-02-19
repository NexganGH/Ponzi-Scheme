from abc import abstractmethod

import numpy as np
from .node import Node
from .node_status import NodeStatus
import json

class Network:

    def __init__(self, n_nodes, capital_per_person=100, ponzi_capital=100, lambda_=0.1, mu=0.1, interest=0.1,
                 interest_calculating_periods=30):
        self.mu = mu
        self.lambda_ = lambda_
        self.ponzi_capital = ponzi_capital
        self.capital_per_person = capital_per_person
        self.interest = interest
        #self.m0 = m0
        #self.m = m
        self.n_nodes = n_nodes
        self.nodes = []
        self.current_size = 0
        self.interest_calculating_periods = interest_calculating_periods
        self.capital_array = None  # Numpy array for fast capital tracking

    def set_parameters(self, parameters):
        self.mu = parameters['mu']
        self.lambda_ = parameters['lambda_']
        self.interest = parameters['interest']
        self.m0 = parameters['m0']
        #self.n_nodes = parameters['n_nodes']
        self.interest_calculating_periods = parameters['interest_calculating_periods']

    @abstractmethod
    def build(self):
        """Metodo da implementare nelle sottoclassi per costruire il network."""
        pass

    def _update_counts(self, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time):
        # Efficiently calculate counts using numpy
        statuses = np.array([node.status for node in self.nodes])
        degrees = np.array([node.degree for node in self.nodes])

        investor_numbers.append(np.sum(statuses == NodeStatus.INVESTOR))
        potential_numbers.append(np.sum(statuses == NodeStatus.POTENTIAL))
        deinvestor_numbers.append(np.sum(statuses == NodeStatus.DEINVESTOR))

        for d in range(degrees_money.shape[0]):
            mask = (degrees == d) #(statuses == NodeStatus.INVESTOR) & (degrees == d)
            if np.any(mask):
                degrees_money[d, time] = self.capital_array[mask].sum() / np.sum(mask)

    def simulate_ponzi(self, max_time_units=10000):
        time = 0
        ponzi = self._ponzi()

        # Logs
        ponzi_capital = [ponzi.capital]
        investor_numbers, potential_numbers, deinvestor_numbers = [], [], []
        degrees_money = np.zeros((50, max_time_units))

        self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)

        last_signal, signal_every = 0, 0.05
        while time < max_time_units and ponzi.capital / self.ponzi_capital >= -10:
            perc = time / max_time_units
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            for i in range(1, len(self.nodes)):  # Skip the Ponzi node
                node = self.nodes[i]

                if node.status == NodeStatus.INVESTOR:
                    if time % self.interest_calculating_periods == 0:
                        interest = self._money_per_turn()
                        self.capital_array[i] += interest
                        ponzi.capital -= interest

                    if np.random.binomial(1, self.mu):  # Node exits
                        exit_capital = self.capital_per_person
                        self.capital_array[i] += exit_capital
                        ponzi.capital -= exit_capital
                        node.status = NodeStatus.DEINVESTOR

                elif node.status == NodeStatus.POTENTIAL:
                    for connection in node.connections:
                        if connection.status == NodeStatus.INVESTOR and np.random.binomial(1, self.lambda_):
                            invest_capital = self.capital_per_person
                            self.capital_array[i] -= invest_capital
                            ponzi.capital += invest_capital
                            node.make_investor(connection)

            # Update logs
            ponzi_capital.append(ponzi.capital)
            self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)
            time += 1

        return [ponzi_capital, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money]

    def _money_per_turn(self):
        return self.interest * self.capital_per_person

    def _ponzi(self) -> Node:
        if len(self.nodes) == 0:
            raise ValueError("Network not yet created.")
        return self.nodes[0]

    def k_distribution(self):
        return np.array([node.k() for node in self.nodes], dtype=int)

    def save_json(self, filename="network.json"):
        """Salva il network in un file JSON."""
        data = {
            "nodes": [
                {
                    "id": i,
                    "status": node.status.name,  # Salviamo il nome dello stato
                    "capital": node.capital,
                    "connections": [self.nodes.index(conn) for conn in node.connections]
                }
                for i, node in enumerate(self.nodes)
            ],
            "params": {
                #"m0": self.m0,
                #"m": self.m,
                "n_nodes": self.n_nodes,
                "capital_per_person": self.capital_per_person,
                "ponzi_capital": self.ponzi_capital,
                "lambda_": self.lambda_,
                "mu": self.mu,
                "interest": self.interest,
                "interest_calculating_periods": self.interest_calculating_periods,
            },
            "model_params": self.get_model_params()
        }


        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Network salvato in {filename}")

    @abstractmethod
    def get_model_params(self):
        print('not implemented')
        pass

    @abstractmethod
    def set_model_params(self, params):
        print('not impleemented')
        pass

    @staticmethod
    def load_json(filename="network.json"):
        """Carica il network da un file JSON e lo ricostruisce."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Creiamo il network con i parametri salvati
        network = Network(**data["params"])
        network.nodes = [Node(node_data["capital"]) for node_data in data["nodes"]]

        network.capital_array = np.full(network.n_nodes, network.capital_per_person, dtype=float)

        # Ripristiniamo gli stati e le connessioni
        for i, node_data in enumerate(data["nodes"]):
            network.nodes[i].status = NodeStatus[node_data["status"]]  # Convertiamo da stringa a enum
            network.nodes[i].connections = [network.nodes[j] for j in node_data["connections"]]

        print(f"Network caricato da {filename}")
        return network