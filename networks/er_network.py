import numpy as np
from networks import Node, Network
from networks.node_status import NodeStatus


class ErNetwork(Network):
    def set_model_params(self, params):
        self.avg_k = params['avg_k']
        #self.p = self.k_avg / (self.n_nodes - 1) if self.n_nodes > 1 else 0

    def get_model_params(self):
        return {'avg_k': self.avg_k}

    def __init__(self, n_nodes, k_avg, capital_per_person=100, ponzi_capital=100, lambda_=0.1, mu=0.1, interest=0.1,
                 interest_calculating_periods=30):
        super().__init__(n_nodes=n_nodes, capital_per_person=capital_per_person,
                         ponzi_capital=ponzi_capital)
        self.avg_k = k_avg
        #self.p = self.k_avg / (self.n_nodes - 1) if self.n_nodes > 1 else 0

    def build(self):
        print('Building ER network...')
        self.nodes = [Node(self.capital_per_person) for _ in range(self.n_nodes)]
        self.capital_array = np.full(self.n_nodes, self.capital_per_person, dtype=float)

        self.nodes[0].capital = self.ponzi_capital
        self.nodes[0].status = NodeStatus.INVESTOR
        self.capital_array[0] = self.ponzi_capital

        p = self.avg_k / (self.n_nodes - 1) if self.n_nodes > 1 else 0

        for i in range(self.n_nodes):
            for j in range(i):
                if np.random.rand() < p:
                    self.nodes[i].connect(self.nodes[j])

        return self
