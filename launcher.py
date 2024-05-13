from Heart_Desease.gradio_gui import Gradio_GUI
from Rhythm_Detection.launcher_rith import launch_rithim


def launch():
    print('Main launcher')
    print('*'*50)
    print('1. Heart Desease')
    print('2. Rithim Detection')
    print('*'*50)
    option = int(input('Select an option: '))

    if option == 1:
        print('Launching Heart Desease Interface...')
        print('\n'*2)
        heart = Gradio_GUI()
        heart.launch_gradio_gui()

    elif option == 2:
        print('Launching Rithim Detection...')
        print('\n'*2)
        launch_rithim()

    