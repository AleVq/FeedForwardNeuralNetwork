import csv

def executeProgram():
    parser = Parser('../risk_factors_cervical_cancer.csv')
    return 0

class Parser:
    def __init__(self, filename):
        self.dataSet = []
        f = open(filename, 'r')#f = open(sys.argv[1], 'rb')
        reader = csv.reader(f)
        for row in reader:
            self.dataSet.append(row)
        for row in self.dataSet:
           for col in row:
               if col == '?':
                   col = -1

executeProgram()