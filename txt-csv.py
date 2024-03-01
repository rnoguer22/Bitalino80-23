import csv
import os
from itertools import islice

class To_csv():
    def __init__(self):
        self.txt_names = [f for f in os.listdir('Data') if f.endswith('.txt')]

    def txt_files_name(self):
        print(self.txt_names)
    
    def to_csv(self):
        for file in self.txt_names:
            with open('Data/' + file, 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                lines = (line.split(",") for line in stripped if line)
                lines_after_4 = islice(lines, 3, None)  # Empieza a partir de la línea 4
                with open('csv/' + file + '.csv', 'w') as out_file:
                    writer = csv.writer(out_file)
                    columnas = self.get_data(file)
                    writer.writerow(columnas)
                    writer.writerows(lines_after_4)

    
    def get_data(self, fichero):
        with open(f'Data/{fichero}', 'r', encoding='utf-8') as file:
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
                return lista

if __name__ == '__main__':
    to_csv = To_csv()
    to_csv.to_csv()