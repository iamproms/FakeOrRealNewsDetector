# ✅ These imports are okay
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ THIS MUST BE THE FIRST Streamlit COMMAND
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# ✅ Only now can you do everything else
nltk.download("stopwords")
nltk.download("wordnet")

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

# ✅ Streamlit UI starts here (no duplicate set_page_config!)
st.title("📰 Fake News Detector")
st.markdown("Analyze news content to determine if it's **REAL** or **FAKE** using an ML model.")

with st.expander("ℹ️ How it works"):
    st.write("This app uses text preprocessing + a trained Naives Bayes model to predict fake news.")

text_input = st.text_area("Paste your article text here:", height=250)

if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("Please enter article text.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(text_input)
            pred = model.predict([cleaned])[0]
            label = "FAKE" if pred == 1 else "REAL"
            st.success(f"Prediction: **{label}**")
            if label == "FAKE":
                st.error("⚠️ This article appears to be **FAKE**.")
            else:
                st.info("✅ This article appears to be **REAL**.")
