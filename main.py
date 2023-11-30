
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
with open('Tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model('Quora_question_pairs.h5')

def predict_duplicate(question1, question2):
    # Tokenize and pad the new sequences
    new_sequences1 = tokenizer.texts_to_sequences([question1])
    new_sequences2 = tokenizer.texts_to_sequences([question2])

    new_padded_sequences1 = pad_sequences(new_sequences1, maxlen=50)
    new_padded_sequences2 = pad_sequences(new_sequences2, maxlen=50)

    # Concatenate the sequences
    new_X = np.hstack([new_padded_sequences1, new_padded_sequences2])

    # Make predictions
    predictions = model.predict(new_X)

    # Threshold predictions to get binary results
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    return binary_predictions.flatten()[0]

def main():
    st.title("Quora Duplicate Question Predictor")

    # User input for new questions
    new_question1 = st.text_input("Enter the first question:")
    new_question2 = st.text_input("Enter the second question:")

    if st.button("Predict"):
        if new_question1 and new_question2:
            # Make predictions
            prediction = predict_duplicate(new_question1, new_question2)

            # Display prediction
            st.write("Prediction:", "Duplicate" if prediction == 1 else "Not Duplicate")
        else:
            st.warning("Please enter both questions.")

if __name__ == '__main__':
    main()
