FinSentiment-Pro
FinSentiment-Pro analyzes financial news sentiment, classifying it as Negative, Neutral, or Positive. It uses a CNN-LSTM model and a Streamlit app for real-time predictions, helping investors and analysts make better decisions.
Overview

Goal: Achieve >95% accuracy with a 5,000-row dataset.
Tools: Python, TensorFlow, Streamlit, NLTK, NLPAug.

Installation

Clone the repo:
bashgit clone https://github.com/A-Square8/FinSentiment-Pro.git
cd FinSentiment-Pro

Set up a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # Use venv\Scripts\activate on Windows

Install dependencies:
bashpip install pandas numpy scikit-learn tensorflow streamlit nlpaug

Add balanced_dataset.csv to data/ and model files (sentiment_model.h5, tokenizer.pkl) to model/.

Usage

Run the app:
bashstreamlit run app.py

Open http://localhost:8501 in your browser.
Enter news text and click "Analyze" to see the sentiment.

Training

Run python train.py to train the model.
Takes 30-60 minutes; saves files in model/.

Files

app.py: Streamlit app.
train.py: Training script.
data/: Datasets.
model/: Model files.

Results

Accuracy: 96.52%
Negative: 0.98 F1-score
Neutral: 0.95 F1-score
Positive: 0.96 F1-score
