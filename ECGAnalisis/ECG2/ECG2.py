import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt




class ECG2:

    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)
    

    def get_columns(self):
        return self.df.columns
    

    #Metodo para aplicar un filtro de media móvil y suavizar la señal y eliminar el ruido
    def data_cleaning(self): 
        window_size = 3
        df_smoothed = self.df.rolling(window=window_size).mean().dropna()
        #Normalizamos los datos para que estén en un rango de 0 a 1
        df_normalized = (df_smoothed - df_smoothed.min()) / (df_smoothed.max() - df_smoothed.min())
        #Eliminamos las señales que de las que no hemos recogido datos
        df_normalized = df_normalized.drop(['I1', 'I2', 'O1', 'O2'], axis=1)
        return df_normalized   


    def find_peaks(self):
        ecg_data = self.data_cleaning()['A2'].values


        # Encuentra los picos R en la señal de ECG
        peaks, _ = find_peaks(ecg_data, height=0)  # Ajusta el valor de altura según tu señal

        # Grafica la señal de ECG y los picos identificados
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, color='blue')
        plt.plot(peaks, ecg_data[peaks], "x", color='red', markersize=10)
        plt.title('Señal de ECG con Picos R Identificados')
        plt.xlabel('Tiempo (muestras)')
        plt.ylabel('Amplitud')
        plt.show()

        # Calcula la frecuencia cardíaca utilizando los picos R identificados
        tiempo_entre_picos = np.diff(peaks)
        frecuencia_cardiaca = 60 / (np.mean(tiempo_entre_picos) / frecuencia_muestreo)  # frecuencia_muestreo es la frecuencia de muestreo de tu señal
        print("Frecuencia Cardíaca:", frecuencia_cardiaca, "latidos por minuto")


ecg2 = ECG2('./csv/ECG2/ECG_Esther_2.csv')
print(ecg2.get_columns())
ecg2.find_peaks()