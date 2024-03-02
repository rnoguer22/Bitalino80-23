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
            #Obtenemos el tipo de archivo, para guardarlo posteriormente en la carpeta correspondiente
            tipo = file[:3]
            if tipo == 'ACC':
                tipo = 'ACCyLUX'
            elif tipo == 'ECG':
                if file[-5] == '2':
                    tipo = 'ECG2'
            #Creamos las carpetas si no existen
            if not os.path.exists('csv/' + tipo):
                os.makedirs('csv/' + tipo)
            with open('Data/' + file, 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                lines = (line.split("\t") for line in stripped if line)
                lines_after_4 = islice(lines, 3, None)  # Empieza a partir de la línea 4
                with open('csv/' + tipo + '/' + file[:-4] + '.csv', 'w', newline='') as out_file:
                    writer = csv.writer(out_file)
                    columnas = self.get_columns(file)
                    writer.writerow(columnas)
                    for line in lines_after_4:
                        writer.writerow(line)

    
    #Metodo para obtener las columnas de los datos
    def get_columns(self, fichero):
        with open(f'Data/{fichero}', 'r', encoding='utf-8') as file:
            data = file.readlines()
            linea = data[1]
            indice_column = linea.find('"column":')
            if indice_column != -1:
                subcadena = linea[indice_column + len('"column":'):]
                indice_corchete_inicio = subcadena.find('[')
                indice_corchete_fin = subcadena.find(']', indice_corchete_inicio)
                lista_subcadena = subcadena[indice_corchete_inicio + 1:indice_corchete_fin]
                lista = lista_subcadena.split(',')
                lista = [elem.strip().strip('"') for elem in lista]
                return lista



if __name__ == '__main__':
    to_csv = To_csv()
    to_csv.to_csv()