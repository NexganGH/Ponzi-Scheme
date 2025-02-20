from abc import abstractmethod

import numpy as np

from .interest_calculator import InterestCalculator
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
        self.capital_per_person = parameters['capital_per_person']
        #self.m0 = parameters['m0']
        #self.n_nodes = parameters['n_nodes']
        self.interest_calculating_periods = parameters['interest_calculating_periods']

    @abstractmethod
    def build(self):
        """Metodo da implementare nelle sottoclassi per costruire il network."""
        pass




    def _money_per_turn(self):
        return self.interest * self.capital_per_person

    def ponzi_node(self) -> Node:
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