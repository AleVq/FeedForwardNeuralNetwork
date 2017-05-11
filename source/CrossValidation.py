import numpy as np
from NeuralNetwork import NeuralNetwork

def cross_validation(ds, targets, learning_rate, k, nodesPerLayer, epoch):
    # each iteration corresponds to a single phase of the k-fold, cross validation
    # each time we take k examples as test-set and the others as the learning-set
    for i in np.arange(k, ds.shape[0]+1, k):
        test_set = np.array(ds[i-k:i,:])
        test_target = np.array(targets[i-k:i])
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
            n.bp_learning(training_set[i], training_target[i], learning_rate, epoch)
        # testing phase
        for i in range(0, test_set.shape[0]):
            if i == 0:
                performance = [n.test(test_set[i], test_target[i])]
            else:
                performance.append(n.test(test_set[i], test_target[i]))
        for i in range(0, len(ds)):
            result = n.test2(ds[i], targets[i])
            print('input:')
            print(ds[i])
            print('target: ')
            print(targets[i])
            print('result:')
            print(result)
            print('\n \n')
