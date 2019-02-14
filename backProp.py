# Author: Jack Wotherspoon
# Created: February 10th, 2019

#import dependencies
import pandas as pd
import numpy as np

#read data
columns=['Refractive_Index','Sodium','Magnesium','Aluminum','Silicon','Potassium','Calcium','Barium','Iron','Glass_Type']
data=pd.read_csv('GlassData.csv', names=columns)
X_data=data.drop('Glass_Type',axis=1)
Y_data=data['Glass_Type']
print(X_data)
print(Y_data)