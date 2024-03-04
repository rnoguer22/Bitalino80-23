import pandas as pd
import matplotlib.pyplot as plt
import os

def clean_ecg_signal(signal, window_size=3):
    #Aplica un suavizado a la señal de ECG usando un promedio móvil
    return signal.rolling(window=window_size).mean().dropna()

def plot_and_save_ecg(filename, save_dir):
    #Genera una gráfica de la señal de ECG y guarda la imagen en el directorio
    df = pd.read_csv(filename)

    ecg_cleaned = clean_ecg_signal(df['A2'])

    ecg_segment = ecg_cleaned.iloc[:1500]

    plt.figure(figsize=(12, 6))
    plt.plot(ecg_segment, color='blue')
    plt.title(f'ECG Clean Signal from {os.path.basename(filename)}')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')

    save_path = os.path.join(save_dir, f'{os.path.basename(filename).replace(".csv", "_clean.png")}')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    ecg_files = ['ECG_Esther.csv', 'ECG_Moyis.csv', 'ECG_Pelayo.csv', 'ECG_Teresa.csv']
    csv_dir = 'csv/ECG'
    save_dir = 'ECGAnalisis/ECG/img'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ecg_file in ecg_files:
        plot_and_save_ecg(os.path.join(csv_dir, ecg_file), save_dir)
