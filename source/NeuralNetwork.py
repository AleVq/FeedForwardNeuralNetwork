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
        # each matrix has m rows, when m = num of previous layer's node
        # and n columns, where n = num of next layer's node
        # values are initiated randomly
        for i in np.arange(1, nodesPerLayer.shape[0]):
            if i == nodesPerLayer.shape[0]-1:
                temp = np.random.random((nodesPerLayer[i - 1] + 1, nodesPerLayer[i]))
            else:
                temp = np.random.random((nodesPerLayer[i-1] + 1, nodesPerLayer[i] + 1))
            self.weight.append(temp)

    # training with back propagation
    # x = current example, t = current target
    def bp_learning(self, x, t, learning_rate):
        # adding the bias to the input layer
        temp = np.concatenate(([1], x))
        a = self.weight[0].T
        layers_output = [temp] # add input layer's output
        # feed forward up to the output layer (included)
        for layer in np.arange(0, len(self.weight)):
            dot_product = np.dot(layers_output[layer], self.weight[layer])
            layers_output.append(np.array(self.activ_func(dot_product)))
            print(layers_output)
        # total error of the net
        E = t - layers_output[-1] # TODO: apply cross entropy function here
        deltas = [E * self.activ_func_der(layers_output[-1])] # start collecting all deltas
        # determine delta for all nodes,
        # from the last hidden layer to the first hidden layer:
        for layer in range(len(layers_output)-2, 0, -1):
            deltas.append(deltas[-1].dot(self.weight[layer].T)
                          * self.activ_func_der(layers_output[layer]))
        # updating weights for each "weight-layer"
        for weight_layer in range(len(self.weights)):
            layer = np.atleast_2d(a[weight_layer])
            delta = np.atleast_2d(deltas[weight_layer])
            self.weights[weight_layer] += learning_rate * layer.T.dot(delta)



def cross_validation(ds, targets, learning_rate, k, nodesPerLayer):
    # each iteration corresponds to a single phase of the k-fold, cross validation
    # each time we take k examples as test-set and the others as the learning-set
    for i in np.arange(k, ds.shape[0]+1, k):
        test_set = np.array(ds[i-k:i,:])
        if i == k:
            training_set = np.array(ds[k:ds.shape[0], :])
            training_target = np.array(targets[k:targets.shape[0]])
        else:
            training_set = np.atleast_2d(ds[0:i-k,:])
            training_set = np.append(training_set, ds[i:ds.shape[0],:], axis=0)
            training_target = np.array(targets[0:i-k])
            training_target = np.append(training_target, targets[i:ds.shape[0]])
        # for each fold of the k-fold we consider a new network to train and test
        # a network with the same topology, but different training set, i.e. different weights
        n= NeuralNetwork(np.array(nodesPerLayer), 'sigmoid')
        # training phase
        for i in range(0,training_set.shape[0]):
            n.bp_learning(training_set[i], training_target[i], learning_rate)
        # testing phase

if __name__ == '__main__':
    # featureSelection
    selector = Selector('../test.csv')
    ds = selector.apply_feature_selection(3)
    # k-fold validation
    cross_validation(ds, selector.targets, 0.003, 2, [3,3,1])

