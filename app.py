import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('model/sentiment_model.h5')
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

label_map = {0: -1, 1: 0, 2: 1}

st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stTextInput > div > div > input {border-radius: 10px; padding: 10px;}
    .stButton > button {background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 20px;}
    .sentiment-negative {color: red; font-weight: bold;}
    .sentiment-neutral {color: gray; font-weight: bold;}
    .sentiment-positive {color: green; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("Financial News Sentiment Analyzer")
st.subheader("Enter financial news text to analyze sentiment")

text_input = st.text_area("Input Text", height=150)

if st.button("Analyze"):
    if text_input:
        sequence = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded)
        pred_label = np.argmax(prediction)
        sentiment = label_map[pred_label]
        
        if sentiment == -1:
            st.markdown('<p class="sentiment-negative">Sentiment: Negative (-1)</p>', unsafe_allow_html=True)
        elif sentiment == 0:
            st.markdown('<p class="sentiment-neutral">Sentiment: Neutral (0)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="sentiment-positive">Sentiment: Positive (1)</p>', unsafe_allow_html=True)
        
        st.write(f"Confidence Scores: Negative: {prediction[0][0]:.2f}, Neutral: {prediction[0][1]:.2f}, Positive: {prediction[0][2]:.2f}")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.caption("Built with Streamlit and TensorFlow | Hybrid CNN-LSTM Model")



