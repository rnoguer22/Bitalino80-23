import os
import gradio as gr

from heart_pred import Heart_Pred



class Gradio_GUI():
        
    #Metodo para devolver el path de la imagen a mostrar
    def select_analysis(self, analysis_type):
        base_path = './img/predictions/'
        if analysis_type == 'Confusion Matrixes':
            analysis = 'confusion_matrixes.png'
        elif analysis_type == 'ROC Curves':
            analysis = 'roc_auc_curve.png'
        else:
            base_path = './img/analysis/'
            if analysis_type == 'Target Distribution':
                analysis = 'target_distribution.png'
            elif analysis_type == 'Distribution of Categorical Features':
                analysis = 'cat_count_plots.png'
            elif analysis_type == 'Distribution of Numerical Features':
                analysis = 'num_density_plots.png'
            elif analysis_type == 'Pairplot of Numerical Features':
                analysis = 'num_pairplot.png'
            elif analysis_type == 'Pearson Heatmap':
                analysis = 'pearson_corr.png'
            elif analysis_type == 'Cramer V Heatmap':
                analysis = 'cramers_corr.png'
            elif analysis_type == 'Reg Plots':
                analysis = 'reg_plots.png'
        return os.path.join(base_path, analysis)
    

    def predict(self, age, sex, thalach, cp, exang, trestbps, oldpeak, fbs, slope, ca, chol, restecg, thal):
        #Realizamos una pequeña transformación de los datos
        if sex == 'Female':
            sex = 0
        else:
            sex = 1
        if exang == 'No':
            exang = 0
        else:
            exang = 1

        heart_pred = Heart_Pred('./csv/heart.csv')
        data = heart_pred.get_data()
        model, scaler = heart_pred.train_random_forest_model(data)
        prediction = heart_pred.predict_target(model, scaler, age, sex, int(cp[0]), trestbps, chol, int(fbs[0]), int(restecg[0]), thalach, exang, oldpeak, int(slope[0]), ca, int(thal[0]))

        if prediction[0] == [0]:
            result = 'Low probability of heart disease'
        elif prediction[0] == [1]:
            result = 'High probability of heart disease'
        else:
            print(prediction[0])
            result = 'Error'
        print(result)

        return gr.Textbox(value=result, visible=True), gr.DataFrame(value=prediction[1], visible=True, label='This is your introduced data:')


    #Metodo para lanzar la interfaz de gradio    
    def launch_gradio_gui(self):
        with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
            gr.Markdown('''
            <h1 style="text-align: left">
            Heart Disease Prediction and Analysis
            </h1>
            ''')
            with gr.Tabs():

                with gr.TabItem('Analysis'):
                    analysis_type = ['Target Distribution', 'Distribution of Categorical Features', 'Distribution of Numerical Features', 'Pairplot of Numerical Features', 'Pearson Heatmap', 'Cramer V Heatmap', 'Reg Plots', 'Confusion Matrixes', 'ROC Curves']
                    dropdown_analyis_type = gr.Dropdown(choices=analysis_type, value=analysis_type[-3], label="Choose the cluster to launch:")
                    text_button = gr.Button("Generate") 
                    analysis_img = gr.Image()
                    text_button.click(self.select_analysis, inputs=dropdown_analyis_type, outputs=analysis_img)

                with gr.TabItem('Predictions'):
                    age = gr.Slider(0, 100, step=1, label='Age:', value=55, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            sex = gr.Radio(['Female', 'Male'], label='Sex', value='Male', interactive=True)
                        thalach = gr.Slider(0, 220, step=1, label='Maximum Heart Rate:', value=111, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            cp_choices = ['0: Typical angina', '1: Atypical angina', '2: Non-anginal pain', '3: Asymptomatic']
                            cp = gr.Dropdown(label='Chest Pain Type:', choices=cp_choices, value=cp_choices[0])
                        exang = gr.Radio(['No', 'Yes'], label='Exercise Induced Angina', value='Yes', interactive=True)
                    with gr.Row():
                        with gr.Column():
                            trestbps = gr.Slider(0, 200, step=1, label='Resting Blood Pressure (mm Hg):', value=140, interactive=True)
                        oldpeak = gr.Slider(0, 10, step=0.1, label='ST Depression Induced by Exercise:', value=5.6, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            fbs_choices = ['0: Lower than 120 mg/dl', '1: Greater than 120 mg/dl']
                            fbs = gr.Dropdown(choices=fbs_choices, label='Fasting Blood Sugar', value=fbs_choices[0], interactive=True)
                        slope_choices = ['0: Upsloping', '1: Flat', '2: Downsloping']
                        slope = gr.Dropdown(label='ST_Slope:', choices=slope_choices, value=slope_choices[0])
                    with gr.Row():
                        with gr.Column():
                            ca = gr.Slider(0, 10, step=1, label='Number of Major Vessels:', value=0, interactive=True)
                        chol = gr.Slider(0, 600, step=1, label='Serum Cholesterol (mg/dl):', value=217, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            restecg_choices = ['0: Normal', '1: ST-T wave abnormality', '2: Left ventricular hypertrophy']
                            restecg = gr.Dropdown(choices=restecg_choices, label='Resting Electrocardiogram (ECG):', value=restecg_choices[1])
                        thal_choices = ['1: Fixed defect', '2: Normal', '3: Reversable defect']
                        thal = gr.Dropdown(choices=thal_choices, label='Thalassemia:', value=thal_choices[2])
                    button = gr.Button("Predict")
                    output_text = gr.Textbox(interactive=True, visible=False)
                    output_df = gr.DataFrame(visible=False)
                    button.click(self.predict, inputs=[age, sex, thalach, cp, exang, trestbps, oldpeak, fbs, slope, ca, chol, restecg, thal], outputs=[output_text, output_df])

        demo.launch(inbrowser=True)
    


if __name__ == '__main__':
    gradio_gui = Gradio_GUI()
    gradio_gui.launch_gradio_gui()