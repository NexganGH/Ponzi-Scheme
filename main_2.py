import matplotlib.pyplot as plt
import numpy as np
from networks import Network, WsNetwork, BaNetwork
from simulation.parameters_calculator import ParameterCalculator
from simulation.finance_data import FinanceData
from simulation import ponzi_simulation


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

parameters = {
    #'m0': 2,
    #'m': 2,
    'n_nodes':30000,
    'interest':0.1/12., # interesse 10% annuo
    'ponzi_capital':5000,
    'lambda_': 0.05/12.,
    'mu': 0.1/12.,
    'interest_calculating_periods':12,
    'capital_per_person': 0
}

net1 = WsNetwork.load_json('my_networks/ws1.json')
net2 = BaNetwork.load_json('my_networks/ba1.json')
list = {'ba1': net2} # 'ws1': net1,

data = FinanceData()
data.download()


interest_calculator = ParameterCalculator(rp= lambda t: 0.1, rr= lambda t: data.market_rr(t) / (1. / 12))
# dobbiamo dividere per 1./12 perché i cambiamenti qui sono scritti su base mensile, invece devono essere scritti annualmente.

for (name, net) in list.items():
    net: Network
    ponzi_simulation = ponzi_simulation.PonziSimulation(
        network=net,
        interest_calculator = interest_calculator,
        max_time_units = 30 * 12,
        dt = 1. / 12,
        lambda_ = lambda t: 0.1,
        mu = lambda t: 0.1,
        capital_per_person = 100,
        ponzi_capital = 100

    )


    ponzi_capital, investor, potential, deinvestor, degrees_money = ponzi_simulation.simulate_ponzi()

    x_vals = np.arange(len(ponzi_capital)) / 12.
    #plt.plot(x_vals, data.interpolated_r_r(x_vals))
    #plt.show()
    print(data.sp500['Close'][0:1])
    # Creazione della figura e asse principale
    fig, ax1 = plt.subplots()

    # Grafico del cambio percentuale (ritorni mensili) sull'asse primario (rosso)
    ax1.plot(x_vals, data.returns_array[:len(x_vals)], label="Monthly Returns (%)", color='tab:red')
    ax1.set_xlabel("Time (Months)")
    ax1.set_ylabel("Monthly Return (%)", color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    #ax1.plot(x_vals, [data.interpolated_r_r(x) for x in x_vals ], color='tab:red', label='test')

    # Creazione di un secondo asse per il prezzo di chiusura dell'S&P 500 e l'investimento iniziale
    ax2 = ax1.twinx()

    # Grafico del prezzo di chiusura dell'S&P 500 (blu)
    ax2.plot(x_vals, data.sp500['Close'][:len(x_vals)], label="S&P 500 Close", color='tab:blue', linestyle='dashed')

    # Calcolo dell'investimento nel tempo
    made_interest = [331.89]
    for i in range(1, len(x_vals)):
        made_interest.append(interest_calculator.ponzi_earnings(made_interest[i - 1], x_vals[i - 1], x_vals[i]))

    # Grafico dell'investimento nel tempo (verde)
    ax2.plot(x_vals, made_interest, label="Investment Growth ($400)", color='tab:green')

    # Etichette per l'asse secondario
    ax2.set_ylabel("S&P 500 Close Price & Investment Value", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Titolo e legenda combinata
    fig.suptitle("S&P 500 Monthly Returns, Closing Price & Investment Growth")

    # Unisce le legende da entrambi gli assi
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # trasformiamo in anni
    #x_vals = np.arange(start=0, stop=len(ponzi_capital)/12., step=1./12)

    ax2.plot(x_vals, np.array(ponzi_capital), label='Capitale di Ponzi (rateo su capitale iniziale)', color='red')
    ax1.plot(x_vals, investor, label='Investors')
    ax1.plot(x_vals, potential, label='Potential')
    ax1.plot(x_vals, deinvestor, label='Deinvestors')


    made_interest = [ponzi_simulation.ponzi_capital]
    for i in range(1, len(x_vals)):
        #print('will calculated', made_interest[i-1], x_vals[i-1], x_vals[i])
        made_interest.append(interest_calculator.ponzi_earnings(made_interest[i - 1], x_vals[i - 1], x_vals[i]))
        #print('calculated ', x_vals[i], made_interest[i])
    print('ending with investors', investor[-1])
    #print(x_vals, made_interest)
    #ax2.plot(x_vals, [interest_calculator.realized_return() for i in range(len(x_vals))])
    ax2.plot(x_vals, made_interest, color='black')
    build_legend(plt)
    plt.savefig(f'imgs/{name}.png')
    plt.show()
