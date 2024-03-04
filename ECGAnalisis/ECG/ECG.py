from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ECG:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        #columns: nSeq,I1,I2,O1,O2,A2
        self.df.columns = ['nSeq','I1','I2','O1','O2','A2']

    def get_names(self):
        return self.data_path.split('/')[-1][:-4]
    
    def plot_peaks(self):
        ecg_data = self.df['A2'].values 
        # Encuentramos los picos R en la señal de ECG
        self.peaks, _ = find_peaks(ecg_data, height=0)

        #Guardamos la grafica del ECG y los picos
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, color='blue')
        plt.plot(self.peaks, ecg_data[self.peaks], "x", color='red', markersize=10)
        plt.title(' {} con Picos R Identificados'.format(self.get_names()))
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.savefig('./ECGAnalisis/ECG/img/{}_peaks.png'.format(self.get_names()))

    def plot_ecg(self):
        ecg_data = self.df['A2'].values 
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, label='ECG')
        plt.title(' {}'.format(self.get_names()))
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.savefig('./ECGAnalisis/ECG/img/{}.png'.format(self.get_names()))

    def get_cardio_freq(self, tiempo):
        num_muestras = self.df.shape[0]
        frecuencia_muestreo = num_muestras / tiempo  
        # Calcula la frecuencia cardíaca utilizando los picos de la señal
        tiempo_entre_picos = np.diff(self.peaks)
        frecuencia_cardiaca = 60 / (np.mean(tiempo_entre_picos) / frecuencia_muestreo)
        return round(frecuencia_cardiaca, 2)

if __name__ == '__main__':
    data_path = './csv/ECG/ECG_Esther.csv'
    ecg = ECG(data_path)
    ecg.plot_peaks()
    frecuencia_cardiaca = ecg.get_cardio_freq(360)
    print(f"Frecuencia cardiaca: {frecuencia_cardiaca} bpm")
    ecg.plot_ecg()