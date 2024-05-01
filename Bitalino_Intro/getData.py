import os
import pandas as pd
import re


class GetData:

    def __init__(self, data_path):
        self.data_path = data_path


    def get_files(self):
        self.files = os.listdir(self.data_path)
        return self.files


    def get_data(self, fichero):
        print(f'{self.data_path}{fichero}')
        with open(f'{self.data_path}{fichero}', 'r', encoding='utf-8') as file:
            data = file.readlines()
            linea = data[1]
            indice_column = linea.find('"column":')

            # Si se encuentra la cadena 'column'
            if indice_column != -1:
                # Extraer la subcadena que comienza después de 'column":'
                subcadena = linea[indice_column + len('"column":'):]

                # Encontrar el índice del primer corchete cuadrado '[' después de 'column":'
                indice_corchete_inicio = subcadena.find('[')

                # Encontrar el índice del último corchete cuadrado ']' después de '['
                indice_corchete_fin = subcadena.find(']', indice_corchete_inicio)

                # Extraer la lista como una subcadena
                lista_subcadena = subcadena[indice_corchete_inicio + 1:indice_corchete_fin]

                # Convertir la subcadena a una lista de elementos
                lista = lista_subcadena.split(',')

                lista = [elem.strip().strip('"') for elem in lista]


data = GetData('./Data/')
for file in data.get_files():
    if file == 'ECG_Esther_2.txt':
        data.get_data(file)