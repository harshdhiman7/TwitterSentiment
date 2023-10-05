#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:18:11 2023

@author: harshdhiman
"""

import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()

loaded_model = pickle.load(open('knn_model.pkl', 'rb'))
tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))


def main():
    st.title('Sentiment Analysis of Tweets with kNN')
    #st.write('This app uses k-Nearest Neighbors (kNN) to classify Iris flowers into three species.')

    # Collect input features from the user
    input_tweet = st.text_input('Enter Tweet',
    'Great feeling to keep scoring and helping the team to move forward in the competition')
    input=tfidf_model.transform([input_tweet])
    
    if st.button('Predict'):
       # Make predictions using the kNN model 
       output=loaded_model.predict(input)
       st.write(f'The predicted sentiment is {output[0]}')
       st.snow() 
   
    

    # Display the prediction
    

if __name__ == '__main__':
    main()
