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
                with open(file + '.csv', 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(('title', 'intro'))
                    writer.writerows(lines_after_4)

if __name__ == '__main__':
    to_csv = To_csv()
    to_csv.to_csv()