import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt




class ECG2:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
    

    def get_columns(self):
        return self.df.columns
    

    def get_name(self):
        return self.data_path[15:-6]
    

    #Metodo para aplicar un filtro de media móvil y suavizar la señal y eliminar el ruido
    def data_cleaning(self): 
        window_size = 3
        df_smoothed = self.df.rolling(window=window_size).mean().dropna()
        #Normalizamos los datos para que estén en un rango de 0 a 1
        df_normalized = (df_smoothed - df_smoothed.min()) / (df_smoothed.max() - df_smoothed.min())
        #Eliminamos las señales que de las que no hemos recogido datos
        df_normalized = df_normalized.drop(['I1', 'I2', 'O1', 'O2'], axis=1)
        return df_normalized   


    #Metodo para graficar la señal de ECG y los picos R identificados
    def plot_peaks(self):
        ecg_data = self.df['A2'].values 
        # Encuentramos los picos R en la señal de ECG
        self.peaks, _ = find_peaks(ecg_data, height=0)  # Ajusta el valor de altura según tu señal
        #Guardamos la grafica del ECG y los picos
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, color='blue')
        plt.plot(self.peaks, ecg_data[self.peaks], "x", color='red', markersize=10)
        plt.title('ECG {} con Picos R Identificados'.format(self.get_name()))
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.savefig('./ECGAnalisis/ECG2/img/{}_ECG_peaks.png'.format(self.get_name()))
    

    #Metodo para obtener la frecuencia cardíaca
    def get_cardio_freq(self, tiempo):
        num_muestras = self.df.shape[0]
        frecuencia_muestreo = num_muestras / tiempo  
        print(frecuencia_muestreo) 
        # Calcula la frecuencia cardíaca utilizando los picos de la señal
        tiempo_entre_picos = np.diff(self.peaks)
        frecuencia_cardiaca = 60 / (np.mean(tiempo_entre_picos) / frecuencia_muestreo)
        return frecuencia_cardiaca




ecg2 = ECG2('./csv/ECG2/ECG_Esther_2.csv')
ecg2.plot_peaks()
print("Frecuencia cardiaca: ", ecg2.get_cardio_freq(21), " latidos por minuto")