import matplotlib.pyplot as plt
import importlib
import networks.network  # Ensure the submodule is directly importable
import numpy as np
from networks import WattsStrogatzNetwork, BaNetwork
from networks.interest_calculator import InterestCalculator
from data import Data

data = Data()
data.download()
interest_calculator = InterestCalculator(r_p = lambda t: 0.1, r_r = lambda t: data.interpolated_r_r(t) / (1./12))

print(interest_calculator.promised_return_at_time(100, 0, 1))

x_vals = np.arange(12*30) / 12.
#plt.plot(x_vals, data.interpolated_r_r(x_vals))
#plt.show()
#print(data.sp500['Close'][0:1])
# Creazione della figura e asse principale
fig, ax1 = plt.subplots()

# Grafico del cambio percentuale (ritorni mensili) sull'asse primario (rosso)
ax1.plot(x_vals, data.returns_array[:len(x_vals)], label="Monthly Returns (%)", color='tab:red')
ax1.plot(x_vals, [data.interpolated_r_r(x) for x in x_vals], color='orange')
#ax1.plot(x_vals, [data.interpolated_r_r(x) for x in x_vals], color='black')
ax1.set_xlabel("Time (Years)")
ax1.set_ylabel("Monthly Return (%)", color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

#ax1.plot(x_vals, [data.interpolated_r_r(x) for x in x_vals ], color='tab:red', label='test')

# Creazione di un secondo asse per il prezzo di chiusura dell'S&P 500 e l'investimento iniziale
ax2 = ax1.twinx()

# Grafico del prezzo di chiusura dell'S&P 500 (blu)
#ax2.plot(x_vals, data.sp500['Close'][:len(x_vals)], label="S&P 500 Close", color='tab:blue', linestyle='dashed')
ax2.plot(x_vals, [5000] + [interest_calculator.promised_return_at_time(5000, x_vals[0], x_vals[i]) for i in range(1, len(x_vals))], label="someone", color='black', linestyle='dashed')

# Calcolo dell'investimento nel tempo
made_interest = [5000]
for i in range(1, len(x_vals)):
    #val = made_interest[i-1]
    #made_interest.append(val + val * data.returns_array[i])
    made_interest.append(interest_calculator.realized_return(made_interest[i - 1], x_vals[i - 1], x_vals[i]))

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