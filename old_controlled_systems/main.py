import matplotlib.pyplot as plt
import importlib
import networks.network  # Ensure the submodule is directly importable
import numpy as np


parameters = {
    'm0': 2,
    'm': 2,
    'n_nodes':30000,
    'interest':0.1/12., # interesse 10% annuo
    'ponzi_capital':5000,
    'lambda_':0.10 / 12.,
    'mu': 0.10 / 12.,
    'interest_calculating_periods':1
}


def build_legend(plt):
    params_str = '\n'.join([f"{key}: {value}" for key, value in parameters.items()])
    plt.legend(
        title="Parametri",
        title_fontsize=13,
        fontsize=11,
        loc="upper right",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.8,
        shadow=True,
        labels=[params_str]  # Inserisce tutti i parametri come unica etichetta
    )




# net = networks.Network(
#     m0=parameters['m0'],
#     m=parameters['m'],
#     n_nodes=parameters['n_nodes'],
#     interest=parameters['interest'], # interesse 10% annuo
#     ponzi_capital=parameters['ponzi_capital'],
#     lambda_=parameters['lambda_'],
#     mu=parameters['mu'],
#     interest_calculating_periods=parameters['interest_calculating_periods']).build()
net = networks.Network.load_json('my_networks/net1.json')

net.set_parameters(parameters)

k_dist = net.k_distribution()
#plt.hist(k_dist[k_dist < 20], range=(0, 20.), )

id_ = 6

#plt.savefig(f'imgs/k_distribution_{id_}')
#plt.show()

print('Average k: ', np.mean(k_dist), ', Average k^2', np.mean(k_dist**2))

ponzi_capital, investor, potential, deinvestor, degrees_money = net.simulate_ponzi(12*30)

#build_legend(plt)
#plt.savefig(f'imgs/ponzi_capital_{id_}')
#plt.show()

#plt.cla()

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# trasformiamo in anni
x_vals = np.arange(start=0, stop=len(ponzi_capital)/12., step=1./12)
ax2.ba_plot(x_vals, np.array(ponzi_capital), label='Capitale di Ponzi (rateo su capitale iniziale)', color='red')
ax1.plot(x_vals, investor, label='Investors')
ax1.plot(x_vals, potential, label='Potential')
ax1.plot(x_vals, deinvestor, label='Deinvestors')
build_legend(plt)
plt.savefig(f'imgs/n_investors_{id_}')
plt.show()

plt.cla()
# Average money for each degree
for d in range(1, len(degrees_money)):
    plt.plot( degrees_money[d], label=f'Degree {d}')

plt.xlim([0, len(ponzi_capital)])
build_legend(plt)

plt.savefig(f'imgs/degrees_money_{id_}')
plt.show()