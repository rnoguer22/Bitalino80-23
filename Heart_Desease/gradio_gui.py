import os
import gradio as gr



class Gradio_GUI():
        
    def select_analysis(self, analysis_type):
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

    
    def launch_gradio_gui(self):
        with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
            gr.Markdown('''
            <h1 style="text-align: left">
            Heart Disease Prediction and Analysis
            </h1>
            ''')
            with gr.Tabs():

                with gr.TabItem('Analysis'):
                    with gr.Row():
                        with gr.Column():
                            analysis_type = ['Target Distribution', 'Distribution of Categorical Features', 'Distribution of Numerical Features', 'Pairplot of Numerical Features', 'Pearson Heatmap', 'Cramer V Heatmap', 'Reg Plots']
                            dropdown_analyis_type = gr.Dropdown(choices=analysis_type, value=analysis_type[0], label="Choose the cluster to launch:")
                            gr.Markdown('<br><br><br><br><br><br>') 
                            text_button = gr.Button("Generate")
                        analysis_img = gr.Image(height=445)
                    text_button.click(self.select_analysis, inputs=dropdown_analyis_type, outputs=analysis_img)

                '''with gr.TabItem('Predictions'):
                    with gr.Row():
                        with gr.Column():
                            dropdown = gr.Dropdown(choices=self.models, value=self.models[0], label="Choose the time series model to visualize the prediction:")
                            text_button = gr.Button("Generate")
                        output_df = gr.DataFrame()
                    text_button.click(self.selection, inputs=dropdown, outputs=output_df)

                with gr.TabItem('ChatBot'):
                    gr.ChatInterface(self.llama3_predict)'''

        demo.launch(inbrowser=True)
    

if __name__ == '__main__':
    gradio_gui = Gradio_GUI()
    gradio_gui.launch_gradio_gui()