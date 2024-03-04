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

    def raw(self):
        self.signals.plot()
    
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
    
    def preprocessing(self):
        
        data = self.data

        #Normalizamos los datos con respecto a la media
        data['A1'] = np.abs(data['A1'] - data['A1'].mean())

        #Eliminamos cualquier posible ruido que pueda haber
        fc = 1000
        fc_norm = fc / (5000 / 2)

        b, a = butter(2, fc_norm, btype='low')
        data['A1'] = lfilter(b, a, data['A1'])

        #Normalizamos los datos con respecto al maximo
        emg_max = data['A1'].max()

        data = data / emg_max * 100

        return data
    
    def group(self, data):
        data_grouped = data.groupby('nSeq').mean()
        return data_grouped

    def plot(self, data):
        data_reset = data.reset_index()
        plt.plot(data_reset['A1'], label='EMG')
        plt.xlabel('Tiempo')
        plt.ylabel('Amplitud')
        plt.title('Señales de EMG de {}'.format(self.nombre))
        plt.legend()
        #plt.savefig('EMGAnalisis/img/EMG_{}_media.png'.format(self.nombre))
        plt.show()



if __name__ == "__main__":
    emg = EMG("Pelayo", "./Data/EMG_Pelayo.txt", "./csv/EMG/EMG_Pelayo.csv")

    

    #Limpieza de datos
    #data_cleaned = emg.clean()
    #outliers = emg.outliers(data_cleaned)
    #emg.preprocessing(outliers)
    

    #Grafico inicial
    emg.preprocessing()

    #Grafico agrupando las frecuencias y tomamdo la media
    #emg.plot(group)

