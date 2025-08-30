import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the balanced dataset
df = pd.read_csv('data/balanced_dataset.csv')
df['label'] = df['label'] + 1  # Map -1, 0, 1 to 0, 1, 2 for model

# Split data
texts = df['text'].values
labels = df['label'].values
_, texts_test, _, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load model and tokenizer
model = load_model('model/sentiment_model.h5')
with open('model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess test data
max_len = 200
sequences_test = tokenizer.texts_to_sequences(texts_test)
X_test = pad_sequences(sequences_test, maxlen=max_len)

# Predict
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(labels_test, pred_labels)
precision = precision_score(labels_test, pred_labels, average='weighted')
recall = recall_score(labels_test, pred_labels, average='weighted')
f1 = f1_score(labels_test, pred_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(labels_test, pred_labels, target_names=['Negative', 'Neutral', 'Positive']))