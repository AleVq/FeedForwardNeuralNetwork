import numpy as np
import math
from featureSelector import Selector

# defining two different activation functions
# sigmoid function and its derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.power(math.e, -x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
# softplus function (an approximation of the relu function)
def softplus(x):
    return np.log(1.0 + np.power(math.e, x))
# softplus' derivative is the sigmoid function
def softplus_deriv(x):
    return sigmoid(x)
def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1.0 - x ** 2
class NeuralNetwork:

    # defining the basic structure of the NN: a list of matrices
    # the i-th matrix represents all weights
    # between the i-th layer and the (i+1)-th layer
    def __init__(self, nodesPerLayer, function): # nodesPerLayer = array in which the i-th element corresponds to the i-th layer
                                        # and gives the number of neurons in that layer
        if function == 'sigmoid':
            self.activ_func = sigmoid
            self.activ_func_der = sigmoid_deriv
        elif function == 'tahn':
            self.activ_func = tanh
            self.activ_func_der = tanh_prime
        else:
            self.activ_func = softplus
            self.activ_func_der = softplus_deriv
        self.weights = []
        # each matrix has m rows, when m = num of previous layer's node
        # and n columns, where n = num of next layer's node
        # values are initiated randomly
        for i in np.arange(1, nodesPerLayer.shape[0]):
            if i == nodesPerLayer.shape[0]-1:
                temp = 2*np.random.random((nodesPerLayer[i - 1] + 1, nodesPerLayer[i]))-1
            else:
                temp = 2*np.random.random((nodesPerLayer[i-1] + 1, nodesPerLayer[i] + 1))-1
            self.weights.append(temp)

    # learning with back propagation
    # x = single example, t = example's target
    def bp_learning(self, X, T, learning_rate, epoch):
        # adding the bias to the input layer
        biases = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((biases.T, X), axis=1)
        for actualEpoch in range(epoch):
            i = np.random.randint(X.shape[0])
            layers_output = [X[i]]
            # feed forward up to the output layer (included)
            for layer in range(len(self.weights)):
                dot_product = np.dot(layers_output[layer], self.weights[layer])
                activation = self.activ_func(dot_product)
                layers_output.append(activation)
            # total error of the net
            E = T[i] - layers_output[-1]
            # starting collecting all deltas grouped by layer
            deltas = [E * self.activ_func_der(layers_output[-1])]
            # determine delta for all nodes,
            # from the last hidden layer to the first one:
            for layer in range(len(layers_output)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[layer].T) *
                              self.activ_func_der(layers_output[layer]))
            # ordering deltas from first hidden layer to output layer
            deltas.reverse()
            # updating weights for each "weight-layer"
            for weight_layer in range(len(self.weights)):
                layer = np.atleast_2d(layers_output[weight_layer])
                delta = np.atleast_2d(deltas[weight_layer])
                self.weights[weight_layer] += learning_rate * layer.T.dot(delta)

    # returns the total error of the network w.r.t. the terget's value
    def test(self, x, target):
        temp = np.concatenate(([1], x))
        layers_output = [temp]
        for weight_layer in range(0, len(self.weights)):
            layers_output = self.activ_func(np.dot(layers_output, self.weights[weight_layer]))
        print('result: ', layers_output[-1], ', error: ', target - layers_output[-1])

if __name__ == '__main__':
    # featureSelection
    selector = Selector('../xorTest.csv')
    ds = selector.apply_feature_selection(2)
    # k-fold validation
    #cross_validation(ds, selector.targets, 0.03, 1, [2,2,1], 1000)
    n = NeuralNetwork(np.array([2,2,1]), 'tah')
    for i in range(0, len(ds)):
        n.bp_learning(ds, selector.targets,0.02, 30000)
    for i in range(0, len(ds)):
        n.test(ds[i], selector.targets[i])