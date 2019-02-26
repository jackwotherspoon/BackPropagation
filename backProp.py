# Author: Jack Wotherspoon
# Student Number: 20012060
# Created: February 10th, 2019

#import dependencies
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#set to true when we want to write to output file
outFile=False

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file: #if file can be read
        csv_reader = reader(file)
        for row in csv_reader: #loop through each row of file and add it
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:                         #for all rows in dataset set each column to a float
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]     #set last column in dataset to integer as these are the classes
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
    stats = [[min(column), max(column)] for column in zip(*dataset)] #loop through columns and grab min and max value
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):                                          #loop through each row but not last column to not mess up classes
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) #equation for minmax data normalization

# Initialize the network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()      #initialize blank list
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]   #add random weight between 0-1 for number of input nodes +1 for bias, loop this for each hidden node so  all inputs are connected to all hidden nodes
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]  #add random weight between 0-1 for number of inputs nodes +1 for bias, loop for number of output nodes so all hidden nodes are connected to all output nodes
    network.append(output_layer)
    #write structural data to output file
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

# Split a dataset into training,validation and test sets
def split_data(dataset):
    dataset_copy = list(dataset)
    #choosing sizes, split is 70% training, 15% validation, 15% test
    train_size=int(len(dataset) * 0.7)      #create 70% training size
    validation_size=int(len(dataset) * 0.15) #create 15% validation size
    test_size=int(len(dataset) * 0.15)       #create 15% testing size
    #initiliaze sets to empy
    train_data=list()
    validation_data=list()
    test_data=list()
    #populate training set, validation set and test set, pop each time so no duplicates
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
        if actual[i] == predicted[i]:  #if actual and predicted are same add 1 to correct
            correct += 1
    return correct / float(len(actual)) * 100.0 #return accuracy

# Calculate neuron activation for an input which is sum of all inputs*weights
def activate(weights, inputs):
    activation = weights[-1]        #activation starts as bias weight
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]    #activation summation function
    return activation

# Transfer neuron activation to node output (this is sigmoid function)
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))   #return the sigmoid function for node output

# Forward propagate input to a network output,work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer.
def forward_propagate(network, row):
    inputs = row
    for layer in network: #loop each layer
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs) #perform activation summation
            neuron['output'] = transfer(activation)          #calculate the neurons output
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)  #derivative of the sigmoid function

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))): #propagate backwards through network
        layer = network[i]
        errors = list()
        if i != len(network)-1:    #if not at last layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])   #update errors as we propagate
                errors.append(error)
        else:
            for j in range(len(layer)):    #if at last layer we calculate new error to propagate next
                neuron = layer[j]
                errors.append(expected[j] - neuron['output']) #calculate error by seeing how close we are to expected
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) #error signal calculated for each neuron is stored with the name ‘delta’

# Update network weights with error
def update_weights(network, row, l_rate, momentum):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]    #get inputs which are output of nodes in layer below
        for neuron in network[i]:      #loop through each node
            for j in range(len(inputs)):    #loop through each weight
                neuron['weights'][j] += (l_rate * neuron['delta'] * inputs[j]) #update weights to allow learning
            neuron['weights'][-1] += momentum * neuron['delta']                #add in momentum to escape local minima and help find global maxima

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, momentum, n_epoch, n_outputs):
    errors=list()
    for epoch in range(n_epoch):    #loop through each epoch
        sum_error=0
        for row in train:       #loop through each row in training set
            outputs = forward_propagate(network, row)       #forward propagate to get outputs
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1                   #number of output values is used to transform class values in the training data into a one hot encoding
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))]) #sum squared error now calculate to monitor learning of training set
            backward_propagate_error(network, expected)     #backpropagate to adjust outputs
            update_weights(network, row, l_rate,momentum)   #update weights as we go
        errors.append(sum_error)
        if epoch > 2 and(errors[epoch-1] - errors[epoch] < 0.01):   #added terminating criteria to break out of epochs if error isnt decreasing fast enough, this avoids overfitting
            print("Terminated training to avoid overfitting at epoch %d" % epoch)
            break
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))  #predicts class based on which output it is closest to

# Backpropagation algorithm with stochastic gradient descent
def back_propagation(train,validate, l_rate, momentum, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1   #number of inputs is how many features we have, take one away which is class
    n_outputs = len(set([row[-1] for row in train]))      #output nodes is how many different values are in the class column
    network = initialize_network(n_inputs, n_hidden, n_outputs) #create network
    train_network(network, train, l_rate, momentum, n_epoch, n_outputs) #train network
    predictions = list()
    for row in validate:    #create predictions for validation set, keeps testing safe
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions, network

#function that predicts class for test set
def testing(test,network):
    predictions = list()
    for row in test:    #seperate function to predict test set when validation set has been deemed sufficient
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions

# Evaluate an algorithm on training set, validation and test set
def evaluate_algorithm(dataset, algorithm,l_rate,momentum,n_epoch,n_hidden):
    train_set, validate_set, test_set = split_data(dataset) #split data accordingly
    scores = list()
    predicted, network = algorithm(train_set,validate_set, l_rate,momentum,n_epoch,n_hidden) #get predictions for validation set and return trained network from backpropagation function
    actual = [row[-1] for row in validate_set]  #grab actual classes of validation set
    accuracy = accuracy_metric(actual, predicted)   #compute accuracy of validation set
    scores.append(accuracy)
    predicted_test=testing(test_set,network)     #predict test set classes
    print(predicted_test)
    actual_test=[row[-1] for row in test_set]   #grab actual test set classes
    print(actual_test)
    accuracy_test=accuracy_metric(actual_test,predicted_test) #accuracy of test set predictions
    scores.append(accuracy_test)
    return scores, network,actual_test,predicted_test

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
scores,network,actual,predicted= evaluate_algorithm(dataset, back_propagation, l_rate,momentum, n_epoch, n_hidden)
print('Validation set Accuracy: %s'% scores[0])
print('Test set Accuracy: %s' % scores[1])
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
#create confusion matrix using sklearn
confusion_mat=confusion_matrix(actual,predicted)
print("Confusion Matrix")
print(confusion_mat)
#create Precision and Recall using sklearn
class_report=classification_report(actual,predicted)
print("Precision and Recall")
print(class_report)
#write assignment requirements to file
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
    f.write("\n\nFinal Weight Vectors:\n")
    f.write(str(network))
    f.write("\nIgnore output and delta part of dictionaries for weights as those were used earlier in code.")
    f.write("\n\nActual Classes: \n")
    f.write(str(actual))
    f.write("\n\nPredicted Classes: \n")
    f.write(str(predicted))
    f.write("\nTest set Accuracy: %s" % scores[1]+"%")
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_mat))
    f.write("\n\nPrecision and Recall: \n")
    f.write(str(class_report))
    f.close()
