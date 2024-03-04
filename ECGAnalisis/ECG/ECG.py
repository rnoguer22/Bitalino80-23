import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def clean_ecg_signal(signal, window_size=3):
    return signal.rolling(window=window_size).mean().dropna()

def calculate_heart_rate(peaks, sampling_rate):
    rr_intervals = np.diff(peaks) / sampling_rate
    heart_rate = 60 / np.mean(rr_intervals)
    return heart_rate

def plot_and_save_ecg(filename, save_dir, sampling_rate=100):
    df = pd.read_csv(filename)
    cleaned_signal = clean_ecg_signal(df['A2'])
    cleaned_signal = cleaned_signal[~cleaned_signal.isna()]  # Eliminar NaN que puedan haber quedado tras el filtro

    # Calcular tamaño de ventana para 15 segundos de señal
    window_size = sampling_rate * 10
    num_samples = len(cleaned_signal)
    num_segments = num_samples // window_size

    if num_segments > 0:
        # Dividir la señal en segmentos
        segments = np.array_split(cleaned_signal, num_segments)
        heart_rates = []

        for i, segment in enumerate(segments):
            peaks, _ = find_peaks(segment, distance=sampling_rate/2)
            heart_rate = calculate_heart_rate(peaks, sampling_rate)
            heart_rates.append(heart_rate)

            # Guardar gráfico de cada segmento
            plt.figure(figsize=(12, 6))
            plt.plot(segment, color='blue')
            plt.plot(peaks, segment.iloc[peaks], "x", color='red', markersize=10)
            plt.title(f'ECG Señal limpia de {os.path.basename(filename)} - Segment {i+1}')
            plt.xlabel('Muestras')
            plt.ylabel('Amplitud')
            plt.savefig(os.path.join(save_dir, f'{os.path.basename(filename).replace(".csv", f"_clean_segment_{i+1}.png")}'))
            plt.close()

        average_heart_rate = np.mean(heart_rates)
        print(f'Frecuencia cardiaca media: {filename}: {average_heart_rate:.2f} latidos por minuto')
    else:
        print(f'No hay suficientes datos {filename} para formar 10 s.')

if __name__ == '__main__':
    ecg_files = ['ECG_Esther.csv', 'ECG_Moyis.csv', 'ECG_Pelayo.csv', 'ECG_Teresa.csv']
    csv_dir = 'csv/ECG'
    save_dir = 'ECGAnalisis/ECG/img'
    sampling_rate = 100  # Hz

    for ecg_file in ecg_files:
        plot_and_save_ecg(os.path.join(csv_dir, ecg_file), save_dir, sampling_rate)

