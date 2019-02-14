# Author: Jack Wotherspoon
# Created: February 10th, 2019

#import dependencies
import pandas as pd
import numpy as np
import random
import math

#read data
columns=['Refractive_Index','Sodium','Magnesium','Aluminum','Silicon','Potassium','Calcium','Barium','Iron','Glass_Type']
data=pd.read_csv('GlassData.csv', names=columns)
X_data=data.drop('Glass_Type',axis=1)
Y_data=data['Glass_Type']
sizeInputs=np.size(X_data,1)

#get data values
X_values=X_data.values
Y_values=Y_data.values
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation using sigmoid function
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


