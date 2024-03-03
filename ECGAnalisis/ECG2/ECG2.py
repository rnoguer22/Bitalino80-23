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


    def plot_peaks(self):
        ecg_data = self.df['A2'].values

        num_muestras = self.df.shape[0]
        tiempo = 21
        frecuencia_muestreo = num_muestras / tiempo  
        print(frecuencia_muestreo)  

        # Encuentra los picos R en la señal de ECG
        peaks, _ = find_peaks(ecg_data, height=0)  # Ajusta el valor de altura según tu señal

        # Grafica la señal de ECG y los picos identificados
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, color='blue')
        plt.plot(peaks, ecg_data[peaks], "x", color='red', markersize=10)
        plt.title('ECG {} con Picos R Identificados'.format(self.get_name()))
        plt.xlabel('Muestras')
        plt.ylabel('Amplitud')
        plt.savefig('./ECGAnalisis/ECG2/img/{}_ECG_peaks.png'.format(self.get_name()))

        # Calcula la frecuencia cardíaca utilizando los picos R identificados
        tiempo_entre_picos = np.diff(peaks)
        frecuencia_cardiaca = 60 / (np.mean(tiempo_entre_picos) / frecuencia_muestreo)  # frecuencia_muestreo es la frecuencia de muestreo de tu señal
        print("Frecuencia Cardíaca:", frecuencia_cardiaca, "latidos por minuto")


ecg2 = ECG2('./csv/ECG2/ECG_Moyis_2.csv')
print(ecg2.get_columns())
ecg2.find_peaks()