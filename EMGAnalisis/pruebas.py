import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from opensignalsreader import OpenSignalsReader

signal = OpenSignalsReader('./Data/EMG_Teresa.txt')

dataP = pd.read_csv('./csv/EMG/EMG_Pelayo.csv', header = 0)
dataE = pd.read_csv('./csv/EMG/EMG_Esther.csv', header = 0)
dataT = pd.read_csv('./csv/EMG/EMG_Teresa.csv', header = 0)
dataM = pd.read_csv('./csv/EMG/EMG_Moyis.csv', header = 0)

def filtrar(data):

    data.drop(columns=['I1', 'I2', 'O1', 'O2'], inplace=True)

    data['A1'] = np.abs(data['A1'] - data['A1'].mean())

    fc = 1000
    fc_norm = fc / (5000 / 2)

    b, a = butter(2, fc_norm, btype='low')
    data['A1'] = lfilter(b, a, data['A1'])

    emg_max = data['A1'].max()

    data = data / emg_max * 100
    return data

dataP = filtrar(dataP)
dataE = filtrar(dataE)
dataT = filtrar(dataT)
dataM = filtrar(dataM)

yP = dataP['A1']
yE = dataE['A1']
yT = dataT['A1']
yM = dataM['A1']

plt.figure(figsize=(13, 5))
plt.plot(yP, label='Pelayo')
#plt.plot(yE, label='Esther')
#plt.plot(yT, label='Teresa')
#plt.plot(yM, label='Moyis')
plt.title('Señal EMG')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.show()