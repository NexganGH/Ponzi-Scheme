from .node_status import NodeStatus

class Node:
    def __init__(self, capital, status=NodeStatus.POTENTIAL):
        self.connections = []  # Use a list for connections
        self.capital = capital
        self.status = status
        self.degree = -1

    def connect(self, node):
        self.connections.append(node)  # Append to the list
        node.connections.append(self)
    def disconnect(self, node):
        self.connections.remove(node)
        node.connections.remove(self)

    def k(self):
        return len(self.connections)  # Degree of the node

    # rendi questo nodo investitore, influenzato dal nodo previous_node
    def make_investor(self, previous_node):
        if previous_node.status != NodeStatus.INVESTOR:
            raise ValueError("Errore, previous_node deve essere un investitore.")
        self.degree = previous_node.degree + 1
        self.status = NodeStatus.INVESTOR