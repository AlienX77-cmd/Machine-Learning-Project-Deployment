# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the trained model
loaded_model = pickle.load(open('C:/Users/Kittipak/Documents/KongData/KU CPE/Year 3 Semester 2/Project/module 5/Deploying Machine Learning Model/diabetes_model.sav', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

# Changing the input_data to numpy array
input_data_numpy = np.asarray(input_data)

# Reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_numpy.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
     
