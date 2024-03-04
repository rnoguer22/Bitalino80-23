import pandas as pd
from scipy.signal import medfilt
from scipy.signal import find_peaks
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

#Funcion para extraccion de características de los picos de la señal EDA
def extraer_caracteristicas_eda(data):
    #picos
    picos, _ = find_peaks(data, height=0.1)#para probar acuerdate de cambiarlo si esta mal
    #caracteristicas de los picos
    numero_picos = len(picos)
    altura_picos = data[picos].mean() if numero_picos > 0 else 0

    return numero_picos, altura_picos

#aplicar extraccion de caracteristicas a cada conjunto de datos
caracteristicas_esther = extraer_caracteristicas_eda(eda_esther['A3_preprocesado'])
caracteristicas_moyis = extraer_caracteristicas_eda(eda_moyis['A3_preprocesado'])
caracteristicas_pelayo = extraer_caracteristicas_eda(eda_pelayo['A3_preprocesado'])
caracteristicas_teresa = extraer_caracteristicas_eda(eda_teresa['A3_preprocesado'])

#visualizar
print(caracteristicas_esther, caracteristicas_moyis, caracteristicas_pelayo, caracteristicas_teresa)

#Aplicar modelo de clasificacion 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#crear conjunto de datos
features = np.array([
    extraer_caracteristicas_eda(eda_esther['A3_preprocesado']),
    extraer_caracteristicas_eda(eda_moyis['A3_preprocesado']),
    extraer_caracteristicas_eda(eda_pelayo['A3_preprocesado']),
    extraer_caracteristicas_eda(eda_teresa['A3_preprocesado'])
])

#etiquetas(simples si no igual no sale)
# 1:(alta reactividad) si numero de picos mayor a 25(ya que hay 2 con 16, 1 con 30 y otro con 142), 0:(baja reactividad)
y = np.array([1 if features[i][0] > 25 else 0 for i in range(len(features))])

#crear modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(features, y)

#prediccion
y_pred = modelo.predict(features)

#evaluar el modelo  
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {accuracy}\n")
print("Classification Report:")
print(report)

#VISUALIZACIÓN RESULTADOS
import matplotlib.pyplot as plt
import seaborn as sns
#preparar datos para visualización
eda_concat = pd.concat([
    eda_esther[['A3_preprocesado']].assign(Subject='Esther'),
    eda_moyis[['A3_preprocesado']].assign(Subject='Moyis'),
    eda_pelayo[['A3_preprocesado']].assign(Subject='Pelayo'),
    eda_teresa[['A3_preprocesado']].assign(Subject='Teresa')
], axis=0).reset_index()

features_df = pd.DataFrame({
    'Subject': ['Esther', 'Moyis', 'Pelayo', 'Teresa'],
    'Number of Peaks': [caracteristicas_esther[0], caracteristicas_moyis[0], caracteristicas_pelayo[0], caracteristicas_teresa[0]],
    'Average Peak Height': [caracteristicas_esther[1], caracteristicas_moyis[1], caracteristicas_pelayo[1], caracteristicas_teresa[1]]
})

#gráfico de series temporales para EDA
plt.figure(figsize=(16, 6))
sns.lineplot(data=eda_concat, x='index', y='A3_preprocesado', hue='Subject', palette='tab10', linewidth=2)
plt.title('EDA Signal Over Time by Subject')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Normalized EDA')
plt.legend(title='Subject')
plt.show()

#histograma de número de picos
plt.figure(figsize=(12, 6))
sns.histplot(data=features_df, x='Number of Peaks', hue='Subject', element='step', palette='tab10', bins=10)
plt.title('Distribution of Number of Peaks')
plt.xlabel('Number of Peaks')
plt.ylabel('Count')
plt.show()

#box plot de altura promedio de los picos
plt.figure(figsize=(12, 6))
sns.boxplot(data=features_df, x='Subject', y='Average Peak Height', palette='tab10')
plt.title('Average Peak Height by Subject')
plt.xlabel('Subject')
plt.ylabel('Average Peak Height')
plt.show()

#scatter plot de número de picos vs altura promedio
plt.figure(figsize=(12, 6))
sns.scatterplot(data=features_df, x='Number of Peaks', y='Average Peak Height', hue='Subject', s=100, palette='tab10')
plt.title('Number of Peaks vs. Average Peak Height')
plt.xlabel('Number of Peaks')
plt.ylabel('Average Peak Height')
plt.legend(title='Subject')
plt.show()