import numpy as np

from networks import Node, Network
from networks.node_status import NodeStatus


class BaNetwork(Network):
    def set_model_params(self, params):
        self.m = params['m']
        self.m0 = params['m0']

    def get_model_params(self):
        return {'m': self.m, 'm0': self.m0}

    def __init__(self, m0, m, n_nodes, capital_per_person=100, ponzi_capital=100, lambda_=0.1, mu=0.1, interest=0.1,
                 interest_calculating_periods=30):
        super().__init__(n_nodes=n_nodes, capital_per_person=capital_per_person,
                         ponzi_capital=ponzi_capital, lambda_=lambda_, mu=mu, interest=interest,
                         interest_calculating_periods=interest_calculating_periods)
        self.m0 = m0
        self.m = m

    def build(self):
        print('Building network...')
        self.nodes = [Node(self.capital_per_person) for _ in range(self.m0)]
        self.capital_array = np.full(self.n_nodes, self.capital_per_person, dtype=float)

        # Fully connect the initial m0 nodes
        for i in range(self.m0):
            for j in range(i):
                self.nodes[i].connect(self.nodes[j])

        # Set Ponzi node specifics
        self.nodes[0].capital = self.ponzi_capital
        self.nodes[0].status = NodeStatus.INVESTOR
        self.nodes[0].degree = 0
        self.capital_array[0] = self.ponzi_capital

        # Add remaining nodes based on preferential attachment

        last_signal, signal_every = 0, 0.05
        while len(self.nodes) < self.n_nodes:
            perc = len(self.nodes) / self.n_nodes
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            degrees = np.array([node.k() for node in self.nodes], dtype=float)
            probs = degrees / degrees.sum()
            choices = np.random.choice(self.nodes, size=self.m, p=probs, replace=False)

            new_node = Node(self.capital_per_person)
            for node in choices:
                new_node.connect(node)
            self.nodes.append(new_node)

        return self