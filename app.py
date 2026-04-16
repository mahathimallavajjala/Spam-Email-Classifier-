import streamlit as st
import pickle
import re
import string

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Enter an email message to check if it's spam or not.")

input_text = st.text_area("✉️ Enter Email Text")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_text = preprocess(input_text)
        vector = vectorizer.transform([clean_text])
        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        if result == 1:
            st.error(f"🚨 Spam Email (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ Not Spam (Confidence: {1-prob:.2f})")