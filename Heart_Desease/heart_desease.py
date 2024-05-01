import pandas as pd

class Heart_Desease():

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
    
    #Metodo para eliminar datos incorrectos (ca y thal)
    def drop_data(self):
        self.data = self.data[self.data['ca'] < 4]
        self.data = self.data[self.data['thal'] > 0]
    
    def rename_columns(self):
        self.data = self.data.rename(
            columns = {'cp':'chest_pain_type', 
                    'trestbps':'resting_blood_pressure', 
                    'chol': 'cholesterol',
                    'fbs': 'fasting_blood_sugar',
                    'restecg' : 'resting_electrocardiogram', 
                    'thalach': 'max_heart_rate_achieved', 
                    'exang': 'exercise_induced_angina',
                    'oldpeak': 'st_depression', 
                    'slope': 'st_slope', 
                    'ca':'num_major_vessels', 
                    'thal': 'thalassemia'})
    

heart_des = Heart_Desease('./Heart_Desease/heart.csv')
heart_des.drop_data()
heart_des.rename_columns()
print(heart_des.data.head())