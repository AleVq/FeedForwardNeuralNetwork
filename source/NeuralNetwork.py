import numpy as np
import math

# defining two different activation functions
# sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))
# softplus function (an approximation of the relu function)
def softplus(x):
    return math.log(1 + math.pow(math.e, x))
# softplus' derivative is the sigmoid function
def softplus_deriv(x):
    return sigmoid(x)

class NeuralNetwork:
    # defining the basic structure of the NN: a list of matrices
    # the i-th matrix represents all weights
    # between the i-th layer and the (i+1)-th layer
    def __init__(self, nodesPerLayer, function): # nodesPerLayer = array in which the i-th element corresponds to the i-th layer
                                        # and gives the number of neurons in that layer
        if function == 'sigmoid':
            self.activ_func = sigmoid
            self.activ_func_der = sigmoid_deriv
        else:
            self.activ_func = softplus
            self.activ_func_der = softplus_deriv
        self.weight = []
        for i in np.arange(1, nodesPerLayer.shape[0]):
            if i == nodesPerLayer.shape[0]-1:
                temp = np.random.random((nodesPerLayer[i - 1] + 1, nodesPerLayer[i]))
            else:
                temp = np.random.random((nodesPerLayer[i-1] + 1, nodesPerLayer[i] + 1))
            self.weight.append(temp)



if __name__ == '__main__':
    n= NeuralNetwork(np.array([2,3,2,1]), 'sigmoid')
    print(n.weight)