from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from scipy.fft import rfft, rfftfreq
import numpy as np


def estimate_heart_rate(ppg_signal):

    smoothed_signal = smooth_signal(ppg_signal)

    heart_rate = calculate_heart_rate(smoothed_signal)

    return heart_rate

def smooth_signal(signal):

    smoothed_signal = signal 
    return smoothed_signal

def calculate_heart_rate(signal):
    
    heart_rate = signal  
    return heart_rate








def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def obtener_señales_interpoladas(roi_image, keypoints):
    signals = []

    for point in keypoints:
        px, py = point.ravel()

        signal_length = 100
        
        signal = np.random.rand(signal_length)

        signals.append(signal)

    return signals


def reduccion_dimensionalidad_PCA(data, n_components):
    pca = PCA(n_components=n_components)
    
    reduced_data = pca.fit_transform(data)
    
    return reduced_data


def calcular_frecuencia_cardiaca(reduced_signals, fs):
    estimated_hr = []

    for signal in reduced_signals:
        dft = rfft(signal)
        psd = np.abs(dft) ** 2
        freqs = rfftfreq(len(signal), 1 / fs)
        idx_max_power = np.argmax(psd)
        dominant_freq_hz = freqs[idx_max_power]
        estimated_hr.append(dominant_freq_hz * 60)

    return max(estimated_hr)