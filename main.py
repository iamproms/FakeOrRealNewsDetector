from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import logging
import os
import time
from bs4 import BeautifulSoup
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn.error")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detector API",
    description="Upload news text and get a FAKE/REAL prediction powered by ML.",
)

# Enable CORS (adjust origins for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at /
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# NLTK setup
try:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

# Load model
model_path = os.getenv("MODEL_PATH", "fake_news_detector_model.joblib")
if not os.path.exists(model_path):
    log.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = joblib.load(model_path)

# Text preprocessing
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(cleaned)

# Schemas
class NewsInput(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: HttpUrl
    follow_redirects: bool = True

# Prediction route
@app.post("/predict")
def predict_news(news: NewsInput):
    try:
        log.info(f"Received text of length {len(news.text)}")
        cleaned = clean_text(news.text)
        time.sleep(1)  # Simulated delay
        pred = model.predict([cleaned])[0]
        label = "FAKE" if pred == 1 else "REAL"
        return {"prediction": label}
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

# Scrape route
@app.post("/scrape")
def scrape_website(req: ScrapeRequest):
    try:
        response = requests.get(req.url, allow_redirects=req.follow_redirects, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n", strip=True)
    return {"text": text}
