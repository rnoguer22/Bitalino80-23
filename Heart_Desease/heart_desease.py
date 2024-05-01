import pandas as pd

class Heart_Desease():

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
    

heart_des = Heart_Desease('./Heart_Desease/heart.csv')
print(heart_des.df.head())