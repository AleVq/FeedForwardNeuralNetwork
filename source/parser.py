import csv
import numpy as np



class Parser:
    def __init__(self, filename):
        temp = []
        f = open(filename, 'r')
        reader = csv.reader(f, delimiter = ';')
        i = 0
        for row in reader:
            if i == 0:
                self.attributes = row
                i = 1
            else:
                temp.append(row)
        self.dataSet = np.asmatrix(temp, dtype=float)
        self.dataSet[self.dataSet==-1] = np.NaN
