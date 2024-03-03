from opensignalsreader import OpenSignalsReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EMG():
    def __init__(self, ruta_txt, ruta_csv):
        self.signals = OpenSignalsReader(ruta_txt)
        self.data = pd.read_csv(ruta_csv, header = 0)

    def get_cols(self):
        return self.data.columns
    
    def clean(self):
        self.data.drop(columns=['I1'], inplace=True)
        self.data.dropna(inplace=True)
        return self.data

    def describe(self):
        mean = self.data.mean()
        std = self.data.std()
        return mean, std
    
    def outliers(self, data):
        mean, std = self.describe()
        threshold = 2*std
        outliers = (data - mean) > (threshold * std)
        data_cleaned = data[~outliers.any(axis=1)]
        return data_cleaned
    
    def group(self, data):
        data_grouped = data.groupby('nSeq').mean()
        return data_grouped
    
    def graph(self):
        self.data.plot()

    def plot(self, data):
        data_reset = data.reset_index()
        plt.plot(data_reset['nSeq'], data_reset['A1'], label='A1_corregido')
        plt.xlabel('nSeq')
        plt.ylabel('Amplitud')
        plt.title('Señales de EMG Limpias')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    emg = EMG("./Data/EMG_Pelayo.txt", "./csv/EMG/EMG_Pelayo.csv")
    data_cleaned = emg.clean()
    outliers = emg.outliers(data_cleaned)
    data = emg.group(outliers)
    emg.plot(data) 

