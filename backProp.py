# Author: Jack Wotherspoon
# Created: February 10th, 2019

#import dependencies
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

#set to true when we want to write to output file
outFile=False

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def split_data(dataset):
    dataset_copy = list(dataset)
    #choosing sizes, split is 70% training, 15% validation, 15% test
    train_size=int(len(dataset) * 0.7)
    validation_size=int(len(dataset) * 0.15)
    test_size=int(len(dataset) * 0.15)
    #initiliaze sets to empy
    train_data=list()
    validation_data=list()
    test_data=list()
    #randomly split data into data set, validation set and test set
    for i in range(train_size):
        index=randrange(len(dataset_copy))
        train_data.append(dataset_copy.pop(index))
    for i in range(validation_size):
        index = randrange(len(dataset_copy))
        validation_data.append(dataset_copy.pop(index))
    for i in range(test_size):
        index = randrange(len(dataset_copy))
        test_data.append(dataset_copy.pop(index))
    return train_data, validation_data, test_data

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm,*args):
    train_set, validate_set, test_set = split_data(dataset)
    scores = list()
    for i in range(2):
        predicted = algorithm(train_set, test_set, *args)
        print(predicted)
        actual = [row[-1] for row in train_set]
        print(actual)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    if outFile:
        f = open('Assignment2_Output.txt', 'w')
        f.write("\n\nInitial Weights:\n")
        f.write(str(network))
    return network

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in train:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions

# Run Backprop on glass dataset
seed(1)
# load and prepare data
filename = 'GlassData.csv'
dataset= load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
l_rate = 0.5
n_epoch = 10
n_hidden = 7
momentum = 0.9 #figure what to do with momentum

scores = evaluate_algorithm(dataset, back_propagation, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
#create output file for Assignment 2:
if outFile:
    f=open('Assignment2_Output.txt','w')
    f.write('\nMean Accuracy: %.3f%%\n' % (sum(scores)/float(len(scores))))
    f.write("Number of Nodes Used: \n")
    f.write("Momentum Parameter Used: "+ str(momentum) +".\n")
    f.write("Regularization Approach Used: \n")
    f.write("Data Preprocessing: \n")
