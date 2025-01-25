import matplotlib.pyplot as plt
import importlib
import networks.network  # Ensure the submodule is directly importable
import numpy as np

importlib.reload(networks.network)  # Reload the submodule
importlib.reload(networks)
net = networks.Network(
    m0=5,
    m=4,
    n_nodes=1000,
    interest=0.5/12, # interesse 50% annuo
    ponzi_capital=5000,
    lambda_=0.005,
    mu=0.0025,
    interest_calculating_periods=15).build()

k_dist = net.k_distribution()

print('Average k: ', np.mean(k_dist), ', Average k^2', np.mean(k_dist**2))

ponzi_capital, investor, potential, deinvestor, degrees_money = net.simulate_ponzi(1000)

plt.plot(np.array(ponzi_capital)/net.ponzi_capital, label='Capitale di Ponzi (rateo su capitale iniziale)')
plt.legend()
plt.show()

plt.cla()

plt.plot(investor, label='Investors')
plt.plot(potential, label='Potential')
plt.plot(deinvestor, label='Deinvestors')
plt.legend()
plt.show()

plt.cla()
# Average money for each degree
for d in range(1, len(degrees_money)):
    plt.plot( degrees_money[d], label=f'Degree {d}')

plt.xlim([0, len(ponzi_capital)])
plt.legend()
plt.show()