import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
classifier = joblib.load('spam_classifier.pkl')

# Assuming you have a label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(['spam', 'ham'])  # Ensure the encoder is fit with the same classes used during training

def classify_email(text):
    # Transform the text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    # Predict using the loaded model
    prediction = classifier.predict(text_vectorized)
    # Convert numerical prediction to label
    label = encoder.inverse_transform(prediction)
    return label[0]

# Streamlit UI
st.title("Email Spam Classifier")
st.write("Enter the email content below to classify it as Spam or Ham:")

# Text area for email input
email_input = st.text_area("Email Content")

if st.button("Classify"):
    if email_input:
        # Classify the email
        result = classify_email(email_input)
        st.write(f"The email is classified as: **{result}**")
    else:
        st.write("Please enter email content to classify.")

