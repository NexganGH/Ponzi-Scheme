import numpy as np
from .node import Node
from .node_status import NodeStatus


class Network:

    def __init__(self, m0, m, n_nodes, capital_per_person=100, ponzi_capital=100, lambda_=0.1, mu=0.1, interest=0.1,
                 interest_calculating_periods=30):
        self.mu = mu
        self.lambda_ = lambda_
        self.ponzi_capital = ponzi_capital
        self.capital_per_person = capital_per_person
        self.interest = interest
        self.m0 = m0
        self.m = m
        self.n_nodes = n_nodes
        self.nodes = []
        self.current_size = 0
        self.interest_calculating_periods = interest_calculating_periods
        self.capital_array = None  # Numpy array for fast capital tracking

    def build(self):
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
        while len(self.nodes) < self.n_nodes:
            degrees = np.array([node.k() for node in self.nodes], dtype=float)
            probs = degrees / degrees.sum()
            choices = np.random.choice(self.nodes, size=self.m, p=probs, replace=False)

            new_node = Node(self.capital_per_person)
            for node in choices:
                new_node.connect(node)
            self.nodes.append(new_node)

        return self

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

        while time < max_time_units and ponzi.capital / self.ponzi_capital >= -1000:
            print(f'At {time}/{max_time_units}')
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
