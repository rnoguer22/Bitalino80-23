from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
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
        heart_des = Heart_Analysis(data_path)
        heart_des.drop_data()
        heart_des.rename_columns()
        heart_des.rename_data()
        self.data = heart_des.get_data()
        seed = 0
        self.names = [
            'Logistic Regression',
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
            "Neural Net"
        ]
        self.classifiers = [
            LogisticRegression(solver="liblinear", random_state=seed),
            KNeighborsClassifier(2),
            SVC(probability=True, random_state=seed),
            NuSVC(probability=True, random_state=seed),
            DecisionTreeClassifier(random_state=seed),
            RandomForestClassifier(random_state=seed),
            AdaBoostClassifier(random_state=seed),
            GradientBoostingClassifier(random_state=seed),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            MLPClassifier(random_state=seed),
        ]


if __name__ == '__main__':
    heart_pred = Heart_Pred('./csv/heart.csv')