from parser import Parser
from featureSelector import Selector
import numpy as np

def executeProgram():
    parser = Parser('../risk_factors_cervical_cancer.csv')
    selector = Selector(parser.dataSet)
    R_1 = Selector.PearsonCoefficient(selector.dataSet, selector.x_mean, selector.y_mean, 0)
    print(R_1)
    return 0


executeProgram()