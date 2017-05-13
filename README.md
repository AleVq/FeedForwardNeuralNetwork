# Feed-forward multi-layered neural network

This is a project given in a *Machine Learning* course.
## Overview
This frameworks aims to implement a machine learning tool. More precisely,
a feed-forward multi-layered neural network which 
- takes a single .csv file that contains the dataset we're interested in, 
- filters the input by applying a feature selection,
- trains by using the backpropagation algorithm,
- gets tested by the k-fold, cross validation method.

## Inputs
 
The .csv is supposed to have a specific structure: the first row must be dedicated to the attributes' labels, 
all values must be both numeric (int or float) and positive, missing values must be represented with the character -1, 
the values separator must be a semicolon, i.e. *';'*. 

The feature selection filters attributes by appying the Pearson's coefficient-based classification. The first *n* attributes are used in the training of the neural network, where *n* is a integer given by the user.

In k-fold cross validation, the dataset is divided in *k* folds, one of which is the test set, where as all the other are the training set. 

## Runs
In the training phase, we present the net one example taken randomly from the training test. This example is feed-fowarded into the net, after getting the result, the error is computed and backpropagated and the weights are updated accordingly. We call this process a *run*. The learning process requires many runs, in the order of magnitude of 10^4.
