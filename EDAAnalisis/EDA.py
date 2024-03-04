import pandas as pd
from scipy.signal import medfilt
import numpy as np

#cargar csv
eda_esther = pd.read_csv('csv/EDA/EDA_Esther.csv')
eda_moyis = pd.read_csv('csv/EDA/EDA_Moyis.csv')
eda_pelayo = pd.read_csv('csv/EDA/EDA_Pelayo.csv')
eda_teresa = pd.read_csv('csv/EDA/EDA_Teresa.csv')

#Funcion para preprocesar los datos (filtrado y normalización)
def preprocesar_eda(data):
    #filtrado de ruido con filtro mediano
    eda_filtrado = medfilt(data['A3'], kernel_size = 5)
    #normalización
    eda_normalizado = (eda_filtrado - np.min(eda_filtrado)) / (np.max(eda_filtrado) - np.min(eda_filtrado))
    
    return eda_normalizado

#aplicar preprocesamiento a cada conjunto de datos
eda_esther['A3_preprocesado'] = preprocesar_eda(eda_esther)
eda_moyis['A3_preprocesado'] = preprocesar_eda(eda_moyis)
eda_pelayo['A3_preprocesado'] = preprocesar_eda(eda_pelayo)
eda_teresa['A3_preprocesado'] = preprocesar_eda(eda_teresa)

#visualizar
print(eda_esther.head(), eda_moyis.head(), eda_pelayo.head(), eda_teresa.head())
