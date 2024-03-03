import pandas as pd



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


ecg2 = ECG2('./csv/ECG2/ECG_Moyis_2.csv')
print(ecg2.get_columns())
print(ecg2.data_cleaning())