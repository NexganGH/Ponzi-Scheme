import yfinance as yf
import numpy as np
import pandas as pd

class Data:
    def __init__(self):
        self.sp500 = None
    def download(self):
        # Scarica i dati mensili dell'S&P 500
        self.sp500 = yf.download('^GSPC', start='1990-01-01', end='2021-01-01', interval='1mo')

        # Calcola i ritorni percentuali mensili
        self.sp500['Monthly Return'] = self.sp500['Close'].pct_change()

        # Rimuove il primo valore NaN
        self.sp500 = self.sp500.dropna()

        # Calcola gli anni trascorsi dal 1990 con precisione sui giorni
        self.sp500['Years Since 1990'] = (self.sp500.index - pd.Timestamp('1990-01-01')).days / 365
        self.sp500['Months Since 1990'] = ((self.sp500.index - pd.Timestamp('1990-01-01')).days / 30).astype(int)

        # Salva i dati nei formati richiesti
        self.months_array = self.sp500['Months Since 1990'].values
        self.returns_array = self.sp500['Monthly Return'].values

        # Crea il dizionario {anni: percent change} usando 'Years Since 1990' per coerenza
        self.returns_dict = dict(zip(self.sp500['Years Since 1990'], self.sp500['Monthly Return']))

        # Stampa i primi 5 valori per verificare
        print(dict(list(self.returns_dict.items())[:5]))

    # Funzione per interpolare i ritorni in base al tempo t (in anni)
    def interpolated_r_r(self, t): # t in years
        #return 0.08
        #return np.interp(t + 1./12, self.sp500['Years Since 1990'], self.sp500['Monthly Return'])
        #print('before', t)
        #t = int(np.ceil(t * 12))  # Convert years to months
        #print('after', t)
        #idx = np.abs(self.months_array - t).argmin()  # Find closest index
        #print('months array is', self.months_array)
        #print('index is ', idx)
        #return self.returns_array[idx]

        #t is in years
        keys_array = np.array(list(self.returns_dict.keys()))  # Converte le chiavi in array NumPy
        idx = np.abs(keys_array - t).argmin()  # Trova l'indice della chiave più vicina
        closest_key = keys_array[min(idx+1, len(keys_array) - 1)]  # Ottieni la chiave effettiva
        if t == 0:
            closest_key = keys_array[0]
        #if t > closest_key:
         #   closest_key = keys_array[idx+1]
        return self.returns_dict[closest_key]  # Restituisce il valore corrispondente
