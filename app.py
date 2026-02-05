import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

st.set_page_config(page_title="Fake News Detector", page_icon="none", layout="centered")

import os

nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    st.error("Missing NLTK stopwords. Please download 'stopwords' manually.")
    st.stop()

try:
    WordNetLemmatizer()
except LookupError:
    st.error("Missing WordNet lemmatizer. Please download 'wordnet' manually.")
    st.stop()


# Setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_model():
    return joblib.load("fake_news_detector_model.joblib")

model = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(cleaned)

st.title("Fake News Detector")
st.markdown("Analyze news content to determine if it is REAL or FAKE using an ML model.")

with st.expander("How it works"):
    st.write("This application uses text preprocessing and a trained Naive Bayes model to predict the authenticity of news articles.")

text_input = st.text_area("Paste your article text here:", height=250)

if st.button("Analyze Content"):
    if not text_input.strip():
        st.warning("Please enter article text.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(text_input)
            pred = model.predict([cleaned])[0]
            label = "FAKE" if pred == 1 else "REAL"
            
            if label == "FAKE":
                st.error("Prediction: This article appears to be FAKE.")
            else:
                st.success("Prediction: This article appears to be REAL.")

st.sidebar.title("About")
st.sidebar.info("This project uses Machine Learning (Naive Bayes) and Natural Language Processing (NLP) to classify news articles.")
