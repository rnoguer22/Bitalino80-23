import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




class Heart_Desease():

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        #Definimos una paleta de colores para usar en los graficos
        self.mypal = ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']
        self.numeric_cols = ['age', 
                            'cholesterol', 
                            'resting_blood_pressure', 
                            'max_heart_rate_achieved', 
                            'st_depression', 
                            'num_major_vessels'] 
        self.categorical_features = ['sex', 
                                      'fasting_blood_sugar', 
                                      'exercise_induced_angina', 
                                      'target', 
                                      'chest_pain_type', 
                                      'resting_electrocardiogram', 
                                      'st_slope', 
                                      'thalassemia']
        warnings.filterwarnings('ignore')

    def get_data(self):
        return self.data
    

    #Metodo para eliminar datos incorrectos (ca y thal)
    def drop_data(self):
        self.data = self.data[self.data['ca'] < 4]
        self.data = self.data[self.data['thal'] > 0]
    

    #Metodo para renombrar las columnas
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
    


    #Metodo para renombrar los datos y que quede mas clara la informacion
    def rename_data(self):
        self.data['sex'][self.data['sex'] == 0] = 'female'
        self.data['sex'][self.data['sex'] == 1] = 'male'

        self.data['chest_pain_type'][self.data['chest_pain_type'] == 0] = 'typical angina'
        self.data['chest_pain_type'][self.data['chest_pain_type'] == 1] = 'atypical angina'
        self.data['chest_pain_type'][self.data['chest_pain_type'] == 2] = 'non-anginal pain'
        self.data['chest_pain_type'][self.data['chest_pain_type'] == 3] = 'asymptomatic'

        self.data['fasting_blood_sugar'][self.data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
        self.data['fasting_blood_sugar'][self.data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

        self.data['resting_electrocardiogram'][self.data['resting_electrocardiogram'] == 0] = 'normal'
        self.data['resting_electrocardiogram'][self.data['resting_electrocardiogram'] == 1] = 'ST-T wave abnormality'
        self.data['resting_electrocardiogram'][self.data['resting_electrocardiogram'] == 2] = 'left ventricular hypertrophy'

        self.data['exercise_induced_angina'][self.data['exercise_induced_angina'] == 0] = 'no'
        self.data['exercise_induced_angina'][self.data['exercise_induced_angina'] == 1] = 'yes'

        self.data['st_slope'][self.data['st_slope'] == 0] = 'upsloping'
        self.data['st_slope'][self.data['st_slope'] == 1] = 'flat'
        self.data['st_slope'][self.data['st_slope'] == 2] = 'downsloping'

        self.data['thalassemia'][self.data['thalassemia'] == 1] = 'fixed defect'
        self.data['thalassemia'][self.data['thalassemia'] == 2] = 'normal'
        self.data['thalassemia'][self.data['thalassemia'] == 3] = 'reversable defect'
    

    def save_describe(self, data):
        df = data.describe().T
        df.to_csv('./csv/summary.csv')



    #Metodo para ver las personas con enfermedades cardiovasculares
    def show_target_distribution(self, data):
        plt.figure(figsize=(7, 5),facecolor='#F6F5F4')
        total = float(len(data))
        #De la columna target, 0 son los que no tienen riesgo de enfermedad cardiovascular y 1 los que si
        ax = sns.countplot(x=data['target'], palette=self.mypal[1::4])
        ax.set_facecolor('#F6F5F4')

        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f} %'.format((height/total)*100), ha="center",
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
            
        ax.set_title('Target variable distribution', fontsize=20, y=1.05)
        sns.despine(right=True)
        sns.despine(offset=5, trim=True)
        plt.savefig('./img/target_distribution.png')


    
    #Metodo para ver las graficas de los datos numericos del dataframe
    def numeric_density_plots(self, data):       
        L = len(self.numeric_cols)
        ncol= 2
        nrow= int(np.ceil(L/ncol))
        #remove_last= (nrow * ncol) - L

        fig, ax = plt.subplots(nrow, ncol, figsize=(16, 14),facecolor='#F6F5F4')   
        fig.subplots_adjust(top=0.92)

        i = 1
        for col in self.numeric_cols:
            plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
            ax = sns.kdeplot(data=data, x=col, hue="target", multiple="stack", palette=self.mypal[1::4]) 
            ax.set_xlabel(col, fontsize=20)
            ax.set_ylabel("density", fontsize=20)
            sns.despine(right=True)
            sns.despine(offset=0, trim=False)
            if col == 'num_major_vessels':
                sns.countplot(data=data, x=col, hue="target", palette=self.mypal[1::4])
                for p in ax.patches:
                        height = p.get_height()
                        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
            i += 1
        plt.suptitle('Distribution of Numerical Features' ,fontsize = 24)
        plt.savefig('./img/num_density_plots.png')


    #Metodo para ver las graficas de las variables categoricas
    def categorical_count_plots(self, data, categorical_features):    
        L = len(categorical_features)
        ncol= 2
        nrow= int(np.ceil(L/ncol))
        remove_last= (nrow * ncol) - L

        fig, ax = plt.subplots(nrow, ncol,figsize=(18, 24), facecolor='#F6F5F4')    
        fig.subplots_adjust(top=0.92)
        ax.flat[-remove_last].set_visible(False)

        i = 1
        for col in categorical_features:
            plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
            ax = sns.countplot(data=data, x=col, hue="target", palette=self.mypal[1::4])
            ax.set_xlabel(col, fontsize=20)
            ax.set_ylabel("count", fontsize=20)
            sns.despine(right=True)
            sns.despine(offset=0, trim=False) 
            plt.legend(facecolor='#F6F5F4')
            
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
            i = i +1
        plt.suptitle('Distribution of Categorical Features' ,fontsize = 24)
        plt.savefig('./img/cat_count_plots.png')
    


    #Sacamos el pairplot para ver la relacion entre las variables
    def numeric_pairplot(self, data):
        cols = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'target']
        sns.pairplot(data[cols], hue="target", corner=True, diag_kind='hist', palette=self.mypal[1::4]);
        plt.suptitle('Pairplot: Numerical Features ' ,fontsize = 24)
        plt.savefig('./img/num_pairplot.png')

    
    #Sacamos las regplots para ver la interrelacion entre las variables
    def reg_plots(self, data):
        fig, ax = plt.subplots(1,4, figsize=(20, 4))
        sns.regplot(data=data[data['target'] ==1], x='age', y='cholesterol', ax = ax[0], color=self.mypal[0], label='1')
        sns.regplot(data=data[data['target'] ==0], x='age', y='cholesterol', ax = ax[0], color=self.mypal[5], label='0')
        sns.regplot(data=data[data['target'] ==1], x='age', y='max_heart_rate_achieved', ax = ax[1], color=self.mypal[0], label='1')
        sns.regplot(data=data[data['target'] ==0], x='age', y='max_heart_rate_achieved', ax = ax[1], color=self.mypal[5], label='0')
        sns.regplot(data=data[data['target'] ==1], x='age', y='resting_blood_pressure', ax = ax[2], color=self.mypal[0], label='1')
        sns.regplot(data=data[data['target'] ==0], x='age', y='resting_blood_pressure', ax = ax[2], color=self.mypal[5], label='0')
        sns.regplot(data=data[data['target'] ==1], x='age', y='st_depression', ax = ax[3], color=self.mypal[0], label='1')
        sns.regplot(data=data[data['target'] ==0], x='age', y='st_depression', ax = ax[3], color=self.mypal[5], label='0')
        plt.suptitle('Reg plots of selected features')
        plt.legend()
        plt.savefig('./img/reg_plots.png')  

    



heart_des = Heart_Desease('./csv/heart.csv')
heart_des.drop_data()
heart_des.rename_columns()
heart_des.rename_data()
data = heart_des.get_data()
#heart_des.save_describe(data)
#heart_des.show_target_distribution(data)
#heart_des.numeric_density_plots(data)
heart_des.categorical_count_plots(data, heart_des.categorical_features[0:-1])
#heart_des.numeric_pairplot(data)
#heart_des.reg_plots(data)
