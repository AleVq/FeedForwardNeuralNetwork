import numpy as np
from NeuralNetwork import NeuralNetwork
from featureSelector import Selector


class CrossValidator:

    # input dataset, ds targets, eta, how many training-validation-test rounds,
    # num. of nodes per layer, max average error in the validation phase
    def __init__(self, learning_rate, rounds, eps):
        self.eta = learning_rate
        self.rounds = rounds
        self.eps = eps
        self.rounds_avg_err = []

    def commence(self, ds, targets, nodes_per_layer):
        # each iteration corresponds to a single phase of the k-fold, cross validation
        # each time we take k examples as test-set, k as validation
        # and the others as the learning-set
        k = int(np.floor(ds.shape[0] / self.rounds))
        for i in np.arange(k, ds.shape[0] + 1, k):
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
            # extracting samples for validation
            valid_set = training_set[0:k,:]
            valid_target= training_target[0:k]
            # for each round of the k-fold we consider a new network to train, validate and test,
            # a network with the same topology, but different training set, i.e. different weights
            n= NeuralNetwork(np.array(nodes_per_layer), 'sigmoid')
            # training phase
            E = 1 # learning and validating until the net's error is as small as we want
            while E > self.eps:
                n.bp_learning(training_set, training_target, self.eta, 5000)
                errors = []
                for j in range(valid_set.shape[0]-1):
                    errors.append(n.test(valid_set[j], valid_target[j]))
                E = np.mean(errors)
            # testing phase
            round_errs = []
            for i in range(0, test_set.shape[0]):
                round_errs.append(n.test(test_set[i], test_target[i]))
            self.rounds_avg_err.append(round_errs)


if __name__ == '__main__':
    # featureSelection
    selector = Selector('../risk_factors_cervical_cancer.csv')
    ds = np.nan_to_num(selector.apply_feature_selection(10))
    # k-fold validation
    cv = CrossValidator(0.002, 10, 0.09)
    cv.commence(ds, selector.targets, [10, 15, 1])
    print(np.array(cv.rounds_avg_err))