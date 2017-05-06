import numpy as np
import math

class Edge:

    def __init__(self):
        self.weight = 0
        self.value = 0

    def updateWeight(self, w):
        self.weight = w

    def setValue(self, x):
        self.value = x

    def get_value(self):
        return self.value


class Neuron:

    def __init__(self):
        self.inputEdges = np.array([])
        self.outputEdges = np.array([])


class inputNeuron(Neuron):

    def __init__(self):
        Neuron.__init__(self)

    def compute_output(self):
        return self.inputEdges[0].get_value()


class HiddenNeuron(Neuron):

    def __init__(self):
        Neuron.__init__(self)

    #def compute_output(self): #sum input weights and values, implementing relu funct