import matplotlib.pyplot as plt
import importlib
import numpy as np
from networks import BaNetwork


parameters = {
    'm0': 5,
    'm': 3,
    'n_nodes': 2000,
    'interest':0.1/12., # interesse 10% annuo
    'ponzi_capital':5000,
    'lambda_':0.025,
    'mu':0.015,
    'interest_calculating_periods':1
}

net = BaNetwork(
    m0=parameters['m0'],
    m=parameters['m'],
    n_nodes=parameters['n_nodes'],
    interest=parameters['interest'], # interesse 10% annuo
    ponzi_capital=parameters['ponzi_capital'],
    lambda_=parameters['lambda_'],
    mu=parameters['mu'],
    interest_calculating_periods=parameters['interest_calculating_periods']).build()

k_dist = net.k_distribution()
plt.hist(k_dist[k_dist < 20], range=(0, 20.), )

net.save_json('my_networks/ba3.json')