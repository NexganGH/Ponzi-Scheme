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
        degrees = np.array([node.degree for node in nodes])

        investor_numbers.append(np.sum(statuses == NodeStatus.INVESTOR))
        potential_numbers.append(np.sum(statuses == NodeStatus.POTENTIAL))
        deinvestor_numbers.append(np.sum(statuses == NodeStatus.DEINVESTOR))

        for d in range(degrees_money.shape[0]):
            mask = (degrees == d) #(statuses == NodeStatus.INVESTOR) & (degrees == d)
            if np.any(mask):
                degrees_money[d, time] = self.network.capital_array[mask].sum() / np.sum(mask)

    def simulate_ponzi(self):
        print(f'Starting simulation')
        time = 0
        ponzi = self.network.ponzi_node()
        nodes = self.network.nodes
        ponzi.capital = self.parameters.starting_capital#self.ponzi_capital


        # Logs
        ponzi_capital_numbers = [ponzi.capital]
        investor_numbers, potential_numbers, deinvestor_numbers = [], [], []
        degrees_money = np.zeros((50, self.max_time_units))

        self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)

        last_signal, signal_every = 0, 0.05
        while (time < self.max_time_units
               #and ponzi.capital / self.ponzi_capital >= -10
               #and (investor_numbers[-1] > 1 or time < 12*10)
                ):
            perc = time / self.max_time_units
            if perc >= last_signal + signal_every:
                print(f'{perc * 100:.2f}% complete')
                last_signal = perc

            if time > 0:
                new_capital = self.interest_calculator.ponzi_earnings(ponzi.capital,
                                                                      (time - 1) * self.dt,
                                                                      time * self.dt)
                #print('updating ponzi capital from ', ponzi.capital, ' to ', new_capital)
                ponzi.capital = new_capital
                #print('updated ponzi capital from ', ponzi_capital[-1], 'to ', ponzi.capital)
            statuses = np.array([node.status for node in nodes])
            for i in range(1, len(nodes)):  # Skip the Ponzi node
                self.evolve_node(i, nodes[i], time)
            statuses = np.array([node.status for node in nodes])

            # Update logs
            ponzi_capital_numbers.append(ponzi.capital)
            self._update_counts(investor_numbers, potential_numbers, deinvestor_numbers, degrees_money, time)
            time += 1

        self.ponzi_capital = ponzi_capital_numbers
        #self.investor_numbers = investor_numbers
        #self.potential_numbers = potential_numbers
        #self.deinvestor_numbers = deinvestor_numbers

        return SimulationResult(
            investor_numbers=investor_numbers,
            potential_numbers=potential_numbers,
            deinvestor_numbers=deinvestor_numbers,
            capital=ponzi_capital_numbers,
            dt=self.dt)
        #return [ponzi_capital, investor_numbers, potential_numbers, deinvestor_numbers, degrees_money]

    def evolve_node(self, i: int, node: Node, time: float):
            ponzi = self.network.ponzi_node()

            if node.status == NodeStatus.INVESTOR:
                # if time % self.interest_calculating_periods == 0:
                #   interest = self._money_per_turn()
                #   self.capital_array[i] += interest
                #  ponzi.capital -= interest

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


