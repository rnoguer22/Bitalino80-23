import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os



class ECG2:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
    

    def get_columns(self):
        return self.df.columns
    

    def get_name(self):
        return self.data_path[15:-6]
    

    def get_num_segments(self, cleaned_signal, seconds=10, sampling_rate=100):
        window_size = sampling_rate * seconds
        num_samples = len(cleaned_signal)
        num_segments = num_samples // window_size
        return num_segments
    

    #Metodo para aplicar un filtro de media móvil, suavizar la señal y eliminar el ruido
    def data_cleaning(self, window_size=3): 
        cleaned_signal = self.df.rolling(window=window_size).mean().dropna()
        return cleaned_signal   


    #Metodo para graficar la señal de ECG y los picos R identificados
    def plot_peaks(self, sampling_rate=100):
        cleaned_signal = self.data_cleaning()['A2'].values

        num_segments = self.get_num_segments(cleaned_signal)

        if num_segments > 0:
            segments = np.array_split(cleaned_signal, num_segments)
            heart_rates = []
            peaks_list = []

            for _, segment in enumerate(segments):
                peaks, _ = find_peaks(segment, distance=sampling_rate/2)
                peaks_list.append(peaks)
                heart_rate = self.get_cardio_freq(peaks, sampling_rate)
                heart_rates.append(heart_rate)
                #Guardamos la grafica del ECG y los picos
                plt.figure(figsize=(12, 6))
                plt.plot(segment, color='blue')
                plt.plot(peaks, segment[peaks], "x", color='red', markersize=10)

            plt.title('ECG {} con Picos R Identificados'.format(self.get_name()))
            plt.xlabel('Muestras')
            plt.ylabel('Amplitud')
            plt.savefig('./ECGAnalisis/ECG2/img/{}_ECG_peaks.png'.format(self.get_name()))
            avg_heart_rate = np.mean(heart_rates)
            print('\nFrecuencia cardiaca media {}: {} latidos por minuto'.format(self.get_name(), avg_heart_rate))
        else:
            print(f'No hay suficientes datos de {self.get_name()} para formar 10 s.')
        return peaks_list

    

    #Metodo para obtener la frecuencia cardíaca
    def get_cardio_freq(self, peaks, sampling_rate=100):
        # Calculamos la frecuencia cardíaca utilizando los picos de la señal
        rr_intervals = np.diff(peaks) / sampling_rate
        frecuencia_cardiaca = 60 / np.mean(rr_intervals)
        return round(frecuencia_cardiaca)


    #Metodo para calcular intervalo QT, onda T y onda P
    def get_params(self, peaks:list, sampling_rate=100):
        qt_intervals = []
        t_wave_durations = []
        p_wave_durations = []
        for peak in peaks:
                for j in range(len(peak) - 1):
                    qt_intervals.append((peak[j + 1] - peak[j]) / sampling_rate)
                    t_wave_durations.append((peak[j + 1] - peak[j]) / sampling_rate)
                    if j > 0:
                        p_wave_durations.append((peak[j] - peak[j - 1]) / sampling_rate)
            
        avg_qt_interval = np.mean(qt_intervals)
        avg_t_wave_duration = np.mean(t_wave_durations)
        avg_p_wave_duration = np.mean(p_wave_durations)
        return round(avg_qt_interval, 6), round(avg_t_wave_duration, 6), round(avg_p_wave_duration, 6)

    
    #Metodo para obtener los resultados de cada usuario
    def get_results(self):
        directory = './csv/ECG2/'
        files = os.listdir(directory)
        for file in files:
            dir_ = directory + file
            ecg2 = ECG2(dir_)
            peaks = ecg2.plot_peaks() 
            qt, t, p = ecg2.get_params(peaks)
            print('Intervalo QT promedio:', qt, 'segundos')
            print('Duración promedio de la onda T:', t, 'segundos')
            print('Duración promedio de la onda P:', p, 'segundos')


if __name__ == '__main__':
    ECG2.get_results(ECG2)