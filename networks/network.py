from abc import abstractmethod

import numpy as np

from .node import Node
from .node_status import NodeStatus
import json

class Network:

    def __init__(self, n_nodes, capital_per_person=100, ponzi_capital=100):
        self.ponzi_capital = ponzi_capital
        self.capital_per_person = capital_per_person
        #self.m0 = m0
        #self.m = m
        self.n_nodes = n_nodes
        self.nodes = []
        self.current_size = 0
        self.capital_array = None  # Numpy array for fast capital tracking

    #def set_parameters(self, parameters):
  #      self.interest_calculating_periods = parameters['interest_calculating_periods']

    @abstractmethod
    def build(self):
        """Metodo da implementare nelle sottoclassi per costruire il network."""
        pass

    def ponzi_node(self) -> Node:
        if len(self.nodes) == 0:
            raise ValueError("Network not yet created.")
        return self.nodes[0]

    def k_distribution(self):
        return np.array([node.k() for node in self.nodes], dtype=int)

    def number_nodes_k(self, k):
        return np.sum(self.k_distribution() == k)
    def print_k_info(self):
        k_dist = self.k_distribution()
        return f'Mean: {np.mean(k_dist)}, k^2: {np.mean(np.square(k_dist))}'

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
            },
            "model_params": self.get_model_params()
        }


        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Network salvato in {filename}")

    @abstractmethod
    def get_model_params(self):
        raise Exception('Not implemented')
        #pass

    @abstractmethod
    def set_model_params(self, params):
        raise Exception('Not implemented')

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