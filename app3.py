#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:18:11 2023

@author: harshdhiman
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#%matplotlib inline
#warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from symspellpy import Verbosity, SymSpell
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer, WordNetLemmatizer
#from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'/Users/harshdhiman/Documents/Codes/MD/Twitter_Data.csv')
df.head()

df=df[:100000:]
df.dropna(inplace=True)
df.category.replace([-1.0,0.0,1.0],['Negative','Neutral','Positive'],inplace=True)



#sns.countplot(df.category)

X = df.clean_text
y = df.category

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

tfidf = TfidfVectorizer()
X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vect,y_train)

knn_pred = knn.predict(X_test_vect)
print(confusion_matrix(y_test,knn_pred))
print(classification_report(y_test,knn_pred))
accuracy= accuracy_score(y_test,knn_pred)


def main():
    st.title('Sentiment Analysis of Tweets with kNN')
    #st.write('This app uses k-Nearest Neighbors (kNN) to classify Iris flowers into three species.')

    # Collect input features from the user
    input_tweet = st.text_input('Enter Tweet',
    'Great feeling to keep scoring and helping the team to move forward in the competition')
    input=tfidf.transform([input_tweet])

    # Create a feature array with the user's input
    output=knn.predict(input)

    # Make predictions using the kNN model
    

    # Display the prediction
    st.write(f'The sentiment of {input_tweet} is : {output}')

if __name__ == '__main__':
    main()



