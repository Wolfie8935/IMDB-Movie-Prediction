import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('./simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3, "?") for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction Function

def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.1 else 'Negative'
    return sentiment

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as a positive or negative')

user_input = st.text_area('Movie Review', placeholder='Write the movie review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    
    pred = 'Positive' if prediction > 0.1 else 'Negative'
    st.write(pred)
else:
    st.write('Please Enter a valid movie review.')