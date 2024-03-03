import pandas as pd



class ECG2:

    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)
    

    def get_columns(self):
        return self.df.columns
    


ecg2 = ECG2('./csv/ECG2/ECG_Moyis_2.csv')
print(ecg2.get_columns())