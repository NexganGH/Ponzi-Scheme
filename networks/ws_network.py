import numpy as np
from .network import Network
from .node import Node
from .node_status import NodeStatus


class WattsStrogatzNetwork(Network):
    def get_model_params(self):
        return {
            'k': self.k,
            'p': self.p
        }
    def set_model_params(self, params):
        self.k = params['k']
        self.p = params['p']

    def __init__(self, n_nodes, k, p, capital_per_person=100, ponzi_capital=100, lambda_=0.1, mu=0.1, interest=0.1,
                 interest_calculating_periods=30):
        super().__init__(n_nodes=n_nodes, capital_per_person=capital_per_person,
                         ponzi_capital=ponzi_capital, interest=interest,
                         interest_calculating_periods=interest_calculating_periods)
        self.k = k  # Ogni nodo inizialmente connesso ai k vicini più prossimi
        self.p = p  # Probabilità di riconnessione

    def build(self):
        print("Building Watts-Strogatz Small-World Network...")
        self.nodes = [Node(self.capital_per_person) for _ in range(self.n_nodes)]
        self.capital_array = np.full(self.n_nodes, self.capital_per_person, dtype=float)

        # Step 1: Creazione della rete regolare (anello con connessioni ai k vicini più prossimi)
        half_k = self.k // 2

        last_signal, signal_every = 0, 0.05


        for i in range(self.n_nodes):
            perc = float(i) / self.n_nodes

            if perc >= last_signal + signal_every:
                print(f'First step, {perc * 100:.2f}% complete')
                last_signal = perc
            for j in range(1, half_k + 1):
                neighbor = (i + j) % self.n_nodes
                self.nodes[i].connect(self.nodes[neighbor])

        # Step 2: Riconnessione casuale degli archi con probabilità p
        last_signal, signal_every = 0, 0.05
        for i in range(self.n_nodes):
            perc = float(i) / self.n_nodes

            if perc >= last_signal + signal_every:
                print(f'Second step, {perc * 100:.2f}% complete')
                last_signal = perc

            for j in range(1, half_k + 1):
                if np.random.rand() < self.p:
                    # Rimuove il vecchio collegamento
                    neighbor = (i + j) % self.n_nodes
                    self.nodes[i].disconnect(self.nodes[neighbor])

                    # Trova un nuovo nodo non ancora connesso
                    possible_nodes = list(set(self.nodes) - set(self.nodes[i].connections) - {self.nodes[i]})
                    if possible_nodes:
                        new_neighbor = np.random.choice(possible_nodes)
                        self.nodes[i].connect(new_neighbor)

        # Imposta il nodo Ponzi
        self.nodes[0].capital = self.ponzi_capital
        self.nodes[0].status = NodeStatus.INVESTOR
        self.capital_array[0] = self.ponzi_capital

        return self
