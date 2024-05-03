import os
import gradio as gr



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
                    age = gr.Slider(0, 100, step=1, label='Age:', value=50, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            sex = gr.Radio(['Female', 'Male'], label='Sex', value='Female', interactive=True)
                        thalach = gr.Slider(0, 220, step=1, label='Maximum Heart Rate:', value=100, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            cp = gr.Dropdown(choices=[])
                        exang = gr.Radio(['No', 'Yes'], label='Exercise Induced Angina', value='No', interactive=True)
                    with gr.Row():
                        with gr.Column():
                            trestbps = gr.Slider(0, 200, step=1, label='Resting Blood Pressure (mm Hg):', value=120, interactive=True)
                        oldpeak = gr.Slider(0, 10, step=0.1, label='ST Depression Induced by Exercise:', value=0, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            fbs = gr.Radio(['<= 120 mg/dl', '> 120 mg/dl'], label='Fasting Blood Sugar', value='<= 120 mg/dl', interactive=True)
                        slope = gr.Dropdown(choices=[])
                    with gr.Row():
                        with gr.Column():
                            ca = gr.Slider(0, 0, step=1, label='Number of Major Vessels:', value=0, interactive=True)
                        chol = gr.Slider(0, 600, step=1, label='Serum Cholesterol (mg/dl):', value=200, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            restecg = gr.Dropdown(choices=[])
                        thal = gr.Dropdown(choices=[])
                    button = gr.Button("Predict")
                    
        demo.launch(inbrowser=True)
    


if __name__ == '__main__':
    gradio_gui = Gradio_GUI()
    gradio_gui.launch_gradio_gui()