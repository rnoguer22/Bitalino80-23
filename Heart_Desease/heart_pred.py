from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
import shap 

from heart_analysis import Heart_Analysis




class Heart_Pred(Heart_Analysis):

    def __init__(self, data_path):
        self.data_path = data_path
        self.seed = 0
    

    def get_pred_names(self):
        names = ['Logistic Regression',
                'Nearest Neighbors',
                'Support Vectors',
                'Nu SVC',
                'Decision Tree',
                'Random Forest',
                'AdaBoost',
                'Gradient Boosting',
                'Naive Bayes',
                'Linear DA',
                'Quadratic DA',
                'Neural Net']
        return names


    def get_pred_classifiers(self):
        classifiers = [
            LogisticRegression(solver="liblinear", random_state=self.seed),
            KNeighborsClassifier(2),
            SVC(probability=True, random_state=self.seed),
            NuSVC(probability=True, random_state=self.seed),
            DecisionTreeClassifier(random_state=self.seed),
            RandomForestClassifier(random_state=self.seed),
            AdaBoostClassifier(random_state=self.seed),
            GradientBoostingClassifier(random_state=self.seed),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            MLPClassifier(random_state=self.seed),
        ]
        return classifiers


    def get_clean_data(self):
        heart_des = Heart_Analysis(self.data_path)
        heart_des.drop_data()
        heart_des.rename_columns()
        heart_des.rename_data()
        return heart_des.get_data()
    

    #Metodo para obtener el train y test de los datos
    def get_train_test(self, data):
        test_size = 0.25
        features = data.columns[:-1]
        X = data[features]
        y = data['target']
        return train_test_split(X, y, test_size = test_size, random_state=self.seed)
    



if __name__ == '__main__':
    heart_pred = Heart_Pred('./csv/heart.csv')
    data = heart_pred.get_clean_data()
    X_train, X_val, y_train, y_val = heart_pred.get_train_test(data)