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

file_path=r'/Users/harshdhiman/Documents/Codes/MD/knn_model.pkl'
file_path2=r'/Users/harshdhiman/Documents/Codes/MD/tfidf_model.pkl'

with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)


with open(file_path2, 'rb') as file2:
    tfidf_model = pickle.load(file2)


def main():
    st.title('Sentiment Analysis of Tweets with kNN')
    #st.write('This app uses k-Nearest Neighbors (kNN) to classify Iris flowers into three species.')

    # Collect input features from the user
    input_tweet = st.text_input('Enter Tweet',
    'Great feeling to keep scoring and helping the team to move forward in the competition')
    input=tfidf_model.transform([input_tweet])

    # Create a feature array with the user's input
    output=loaded_model.predict(input)

    # Make predictions using the kNN model
    

    # Display the prediction
    st.write(f'The sentiment of {input_tweet} is : {output}')

if __name__ == '__main__':
    main()