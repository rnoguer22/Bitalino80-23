from opensignalsreader import OpenSignalsReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

class EMG():
    def __init__(self, nombre, ruta_txt, ruta_csv):
        self.signals = OpenSignalsReader(ruta_txt)
        self.data = pd.read_csv(ruta_csv, header = 0)
        self.nombre = nombre

    def get_cols(self):
        return self.data.columns
    
    def clean(self):
        self.data.drop(columns=['I1', 'I2', 'O1', 'O2'], inplace=True)
        self.data.dropna(inplace=True)
        return self.data

    def describe(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std
    
    def outliers(self, data):
        mean, std = self.describe(data)
        threshold = 2*std
        outliers = (data - mean) > (threshold * std)
        data_cleaned = data[~outliers.any(axis=1)]
        return data_cleaned
    
    def noise(self, data):
        fc = 1000
        fc_norm = fc / (10000 / 2)

        b, a = butter(4, fc_norm, btype='low')
        data['A1'] = lfilter(b, a, data['A1'])
        return data
    
    def group(self, data):
        data_grouped = data.groupby('nSeq').mean()
        return data_grouped
    
    def graph(self):
        self.signals.plot()
        plt.savefig('EMGAnalisis/img/EMG_{}.png'.format(self.nombre))
        

    def plot(self, data):
        data_reset = data.reset_index()
        plt.plot(data_reset['nSeq'], data_reset['A1'], label='EMG_grouped')
        plt.xlabel('nSeq')
        plt.ylabel('Amplitud')
        plt.title('Señales de EMG Limpias')
        plt.legend()
        plt.savefig('EMGAnalisis/img/EMG_{}_media.png'.format(self.nombre))
        plt.show()



if __name__ == "__main__":
    emg = EMG("Pelayo", "./Data/EMG_Pelayo.txt", "./csv/EMG/EMG_Pelayo.csv")

    #Limpieza de datos
    data_cleaned = emg.clean()
    outliers = emg.outliers(data_cleaned)
    group = emg.group(outliers)

    #Grafico inicial
    emg.graph()

    #Grafico agrupando las frecuencias y tomamdo la media
    #emg.plot(group)

