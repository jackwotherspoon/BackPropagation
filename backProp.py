# Author: Jack Wotherspoon
# Created: February 10th, 2019

#import dependencies
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

#set to true when we want to write to output file
outFile=True #set to true when we want to write to file

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
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    if outFile:
        f.write("\nStructure of network is (%d,%d,%d)" % (n_inputs,n_hidden,n_outputs))
        f.write("\nAs seen 9 input nodes were used, one for each feature given in CSV file.\n")
        f.write("The number of hidden layers is 1 and it has 8 hidden nodes. \n")
        f.write("Only one hidden layer was chosen through trial and error of viewing accuracy on validation set when changing number of layers. Adding a second layer slightly increased accuracy but greatly increased computation and thus was rejected.")
        f.write("\nThe 8 hidden nodes was determined through same trial and error of viewing accuracy on validation set. The 8 nodes gave best results.")
        f.write("\nThere are 6 output nodes, this was chosen due to the number of glass types given which is 6.\n")
        f.write("\nInitial Weights:\n")
        f.write("Input to hidden layer weights:" +str(hidden_layer)+"\n")
        f.write("Hidden to output layer weights:" +str(output_layer))
        f.write("\n\nNode Output Function Used: ")
        f.write("\nOutput node function used for this assignment is sigmoid function as it allows for a smooth and bounded function for the total input. It also has the added benefit of having nice derivatives which make learning the weights of a neural network easier.")
    return network

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

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation to node output
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
def update_weights(network, row, l_rate, momentum):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += (l_rate * neuron['delta'] * inputs[j])
            neuron['weights'][-1] += momentum * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, momentum, n_epoch, n_outputs):
    errors=list()
    for epoch in range(n_epoch):
        sum_error=0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate,momentum)
        errors.append(sum_error)
        if epoch > 2 and(errors[epoch-1] - errors[epoch] < 0.01):
            print("Terminated training to avoid overfitting at epoch %d" % epoch)
            break
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train,validate, l_rate, momentum, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, momentum, n_epoch, n_outputs)
    predictions = list()
    for row in validate:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions, network

#function that predicts class for test set
def testing(test,network):
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions

# Evaluate an algorithm on training set, validation and test set
def evaluate_algorithm(dataset, algorithm,l_rate,momentum,n_epoch,n_hidden):
    train_set, validate_set, test_set = split_data(dataset)
    scores = list()
    predicted, network = algorithm(train_set,validate_set, l_rate,momentum,n_epoch,n_hidden)
    #print(predicted)
    actual = [row[-1] for row in validate_set]
    #print(actual)
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    predicted_test=testing(test_set,network)
    actual_test=[row[-1] for row in test_set]
    accuracy_test=accuracy_metric(actual_test,predicted_test)
    scores.append(accuracy_test)
    return scores, network

# Run Backprop on glass dataset
# load and prepare data
filename = 'GlassData.csv'
dataset= load_csv(filename)
#convert csv strings to floats
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables, this is data preprocessing
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# key architecture
l_rate = 0.2
n_epoch = 1000
n_hidden = 8
momentum = 0.9

#create output file and write to it
if outFile:
    f=open('Assignment2_Output.txt','w')
    f.write("Assignment 2 Output File \n")
    f.write("\nStudent: Jack Wotherspoon \n")
    f.write("Student Number: 20012060 \n")
scores,network= evaluate_algorithm(dataset, back_propagation, l_rate,momentum, n_epoch, n_hidden)
print(network['weights'])
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
if outFile:
    f.write("\n\nLearning rate: %.1f" % l_rate)
    f.write("\nLearning rate was chosen to be 0.2 by using it as a parameter for my validation testing. Through many trials and iterations it was determined to be the best for accuracy. ")
    f.write("\n\nTerminating Critera: ")
    f.write("\nTerminating criteria is one of two things. Training stops when the 1000 epochs are completed or when sum-squared error on training data begins to change super slowly. ")
    f.write("\nThe number of epochs was used as a parameter for validation set, 1000 was determined to be the cutoff as to where increasing the number of epochs seeemed to no longer improve accuracy by a noticeable amount.")
    f.write("\nSum-squared error is checked each epoch and if the change in error from one epoch to the next is less than 0.01 than it stops training. This was a parameter for validation testing and was determined through trials and iterations.")
    f.write("\nThis terminating criteria allows the training to not overfit the data by stopping it once it has learned sufficiently.")
    f.write("\n\nMomentum value: %.1f" % momentum)
    f.write("\nThis momentum value was determined through using it as a validation parameter. It allowed my training accuracy to go from 65% to roughly 80% on my training data, probably escaping a local minima. This then transfered to a better validation accuracy. ")
    f.write("\n\nData Preprocessing:")
    f.write("\nThe first part of my preprocessing was to convert the CSV input data from strings to floats and the class column from string to integer. This allows for much easier usage of the data further along.")
    f.write("\nThe main preprocessing I did was normalizing the data. Since the raw data for features vary in scale, some are in range 0-1 and some as high as being in the seventies. This would cause the feature with values in the seventies to alter the node outputs way more than the others.")
    f.write("\n Normalizing the data fixes this. I normalized my data into the range of 0-1 as it is good practice to normalize input values to the range of the transfer function which is sigmoid and outputs between 0-1.")
    f.write("\n\nData Splitting")
    f.write("\nAs seen in my split_data() function I chose to split my data into 70% training data, 15% validation data, and 15% testing data.")
    f.write("\nI chose these splits because you always want the majority of data to be in the training set where it can learn and update weights and biases.")
    f.write("\nMy validation set was given 15% in order to allow me to tune parameters of my network. Using my validation set I was able to tune my number of hidden layers and nodes, learning rate, number of epochs, and terminating criteria and improve validation accuracy with each.")
    f.write("\nThe remaining 15% was for the test set which allowed for an unbiased evaluation of the final model. Only used once the validation set was sufficiently trained (using training and validation set).")
    f.close()
