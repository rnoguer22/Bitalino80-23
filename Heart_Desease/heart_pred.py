from heart_analysis import Heart_Analysis

class Heart_Pred(Heart_Analysis):

    def __init__(self, data_path):
        self.data_path = data_path
        heart_des = Heart_Analysis(data_path)
        heart_des.drop_data()
        heart_des.rename_columns()
        heart_des.rename_data()
        data = heart_des.get_data()
        print(data.head())


if __name__ == '__main__':
    heart_pred = Heart_Pred('./csv/heart.csv')