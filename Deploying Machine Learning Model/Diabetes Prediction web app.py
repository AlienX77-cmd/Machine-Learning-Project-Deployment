# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:11:14 2023

@author: Kittipak
"""

import numpy as np
import pickle
import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open('C:/Users/Kittipak/Documents/KongData/KU CPE/Year 3 Semester 2/Project/module 5/Deploying Machine Learning Model/diabetes_model.sav', 'rb'))

# Creating a function for Prediction System

def diabetes_prediction(input_data):
    
    # Changing the input_data to numpy array
    input_data_numpy = np.asarray(input_data)

    # Reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_numpy.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
       
if __name__ == '__main__':
    main()
    
