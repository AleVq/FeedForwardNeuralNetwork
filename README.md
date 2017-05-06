# FeedForwardNeuralNetwork
Feed forward multi layered neural network

This frameworks aims to implement a machine learning tool. More precisely,
a feed-forward multi-layered neural network which 
- takes a single .csv file that contains the dataset we're interested in, 
- filters the input by applying a feature selection,
- trains by using the backpropagation algorithm,
- gets tested by the k-fold, cross validation method.

The .csv is supposed to have a specific structure: the first row must be dedicated to the attributes' labels, 
all values must be both numeric (int or float) and positive, missing values must be represented with the character -1, 
the values separator must be a semicolon, i.e. *';'*. 
