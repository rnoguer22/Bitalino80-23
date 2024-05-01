import pandas as pd

class Heart_Desease():

    def __init__(self, data_path):
        self.data_path = data_path
    
    def get_df(self):
        df = pd.read_csv(self.data_path)
        return df
    
    #Metodo para eliminar datos incorrectos (ca y thal)
    def drop_data(self, df):
        df = df[df['ca'] < 4]
        df = df[df['thal'] > 0]
        return df
    

heart_des = Heart_Desease('./Heart_Desease/heart.csv')
initial_df = heart_des.get_df()
clean_df = heart_des.drop_data(initial_df)
print(len(clean_df))