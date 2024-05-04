from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


    def get_data(self):
        heart_des = Heart_Analysis(self.data_path)
        return heart_des.get_data()
    

    #Metodo para obtener el train y test de los datos
    def get_train_test(self, data):
        test_size = 0.25
        features = data.columns[:-1]
        X = data[features]
        y = data['target']
        return train_test_split(X, y, test_size = test_size, random_state=self.seed)


    def score_summary(self, names, classifiers, X_train, X_val, y_train, y_val):
        '''
        Given a list of classiers, this function calculates the accuracy, 
        ROC_AUC and Recall and returns the values in a dataframe
        '''
        
        cols = ["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
        data_table = pd.DataFrame(columns=cols)
        
        for name, clf in zip(names, classifiers):        
            clf.fit(X_train, y_train)
            
            pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, pred)

            pred_proba = clf.predict_proba(X_val)[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
            roc_auc = auc(fpr, tpr)
            
            cm = confusion_matrix(y_val, pred) 
            
            #Recall = TP/(TP+FN)
            recall = cm[1,1]/(cm[1,1] +cm[1,0])
            
            #Precision = TP/(TP+FP)
            precision = cm[1,1]/(cm[1,1] +cm[0,1])
            
            #F1 score = TP/(TP+FP)
            f1 = 2*recall*precision/(recall + precision)

            df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1]], columns=cols)
            data_table = data_table._append(df)     
            
        return(np.round(data_table.reset_index(drop=True), 2))



    def plot_conf_matrix(self, names, classifiers, nrows, ncols, fig_a, fig_b, X_train, X_val, y_train, y_val):
        '''
        Plots confusion matrices in subplots.
        
        Args:
            names : list of names of the classifiers
            classifiers : list of classification algorithms
            nrows, ncols : number of rows and columns in the subplots
            fig_a, fig_b : dimensions of the figure size
        '''
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_a, fig_b))
        i = 0
        for clf, ax in zip(classifiers, axes.flatten()):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            cm = confusion_matrix(y_val, y_pred) 
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)  
            disp.plot(ax=ax)
            ax.set_title(names[i])
            i += 1
        plt.tight_layout()
        plt.savefig('./img/predictions/confusion_matrixes.png')

        
        
    def roc_auc_curve(self, names, classifiers, X_train, X_val, y_train, y_val):
        '''
        Given a list of classifiers, this function plots the ROC curves

        '''       
        plt.figure(figsize=(12, 8))   
            
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            pred_proba = clf.predict_proba(X_val)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=3, label= name +' ROC curve (area = %0.2f)' % (roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic (ROC) curves', fontsize=20)
            plt.legend(loc="lower right")       
        plt.savefig('./img/predictions/roc_auc_curve.png')



    def train_random_forest_model(self, data):
        X = data.drop(columns=['target'])
        y = data['target']
        #Escalamos los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #Entrenamos el Random Forest
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X_scaled, y)
        return model, scaler



    def predict_target(self, model, scaler, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        '''print("Introduce los valores de las características:")
        age = int(input("Edad: "))
        sex = int(input("Sexo (0 para mujer, 1 para hombre): "))
        cp = int(input("Tipo de dolor en el pecho (0-3): "))
        trestbps = int(input("Presión arterial en reposo (mm Hg): "))
        chol = int(input("Colesterol sérico (mg/dl): "))
        fbs = int(input("Nivel de azúcar en sangre en ayunas (0 para <= 120 mg/dl, 1 para > 120 mg/dl): "))
        restecg = int(input("Resultado del electrocardiograma en reposo (0-2): "))
        thalach = int(input("Frecuencia cardíaca máxima alcanzada: "))
        exang = int(input("Angina inducida por el ejercicio (0 para no, 1 para sí): "))
        oldpeak = float(input("Depresión del segmento ST inducida por el ejercicio en relación con el reposo: "))
        slope = int(input("La pendiente del segmento ST de ejercicio máximo (0-2): "))
        ca = int(input("Número de vasos principales coloreados por flourosopía: "))
        thal = int(input("Resultado de la prueba de esfuerzo cardíaco (0-3): "))'''

        #Creamos un dataframe con los datos introducidos por el usuario
        data = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
                             'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
                             'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]})
        #Escalamos los datos
        data_scaled = scaler.transform(data)
        print(model.predict_proba(data_scaled))
        #Realizamos la prediccion
        prediction = model.predict(data_scaled)
        #Mostramos el resultado
        return prediction, data
    



'''if __name__ == '__main__':
    heart_pred = Heart_Pred('./csv/heart.csv')
    data = heart_pred.get_data()
    names = heart_pred.get_pred_names()
    classifiers = heart_pred.get_pred_classifiers()
    X_train, X_val, y_train, y_val = heart_pred.get_train_test(data)
    df = heart_pred.score_summary(names, classifiers, X_train, X_val, y_train, y_val)
    print(df)
    df.to_csv('./csv/predictions/predictions.csv', index=False)
    heart_pred.roc_auc_curve(names, classifiers, X_train, X_val, y_train, y_val)
    heart_pred.plot_conf_matrix(names, classifiers, nrows=4, ncols=3, fig_a=12, fig_b=12, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    model, scaler = heart_pred.train_random_forest_model(data)
    print(heart_pred.predict_target(model, scaler))'''