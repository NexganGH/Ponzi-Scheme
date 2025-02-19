import matplotlib.pyplot as plt
import importlib
import networks.network  # Ensure the submodule is directly importable
import numpy as np
from networks import WattsStrogatzNetwork

parameters = {
    'k': 10,
    'p': 0.2,
    'n_nodes':50000,
    'interest':0.1/12., # interesse 10% annuo
    'ponzi_capital':5000,
    'lambda_':0.025,
    'mu':0.015,
    'interest_calculating_periods':1
}

net = WattsStrogatzNetwork(
    k=parameters['k'],
    p=parameters['p'],
    n_nodes=parameters['n_nodes'],
    interest=parameters['interest'], # interesse 10% annuo
    ponzi_capital=parameters['ponzi_capital'],
    lambda_=parameters['lambda_'],
    mu=parameters['mu'],
    interest_calculating_periods=parameters['interest_calculating_periods']).build()

net.save_json('my_networks/ws2.json')
k_dist = net.k_distribution()
plt.hist(k_dist[k_dist < 20], range=(0, 20.), )
plt.show()
