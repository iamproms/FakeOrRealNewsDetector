# Fake News Detector

An AI-powered application for detecting fake news using machine learning. This project provides a robust framework for analyzing news articles and determining their authenticity through natural language processing.

## Table of Contents
- Introduction
- Features
- Technology Stack
- Installation
- Usage
- API Endpoints
- Dataset
- License

## Introduction
The Fake News Detector is designed to combat misinformation by providing users with a tool to verify the credibility of news content. It utilizes a trained Naive Bayes model to classify news as either REAL or FAKE based on the textual content.

## Features
- Real-time News Analysis: Paste article text to get immediate classification.
- URL Scraping: Automatically extract and analyze content from news websites.
- Dual Interfaces: Choose between a professional web interface or an interactive Streamlit dashboard.
- High Performance: Fast inference using a pre-trained scikit-learn model.

## Technology Stack
- Backend: FastAPI
- Frontend: Vanilla CSS, JavaScript, Materialize CSS
- Dashboard: Streamlit
- Machine Learning: Scikit-learn, NLTK, Joblib
- Scraping: BeautifulSoup4, Requests

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamproms/FakeOrRealNewsDetector.git
   cd FakeOrRealNewsDetector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: If you plan to use the Streamlit interface, also install streamlit:
   ```bash
   pip install streamlit
   ```

## Usage

### Running the FastAPI Server
To start the backend and the main web interface:
```bash
uvicorn main:app --reload
```
Once running, navigate to http://127.0.0.1:8000 in your browser.

### Running the Streamlit Dashboard
To launch the interactive dashboard:
```bash
streamlit run app.py
```

## API Endpoints

### POST /predict
Analyzes text content and returns a prediction.
- Payload: `{"text": "Your article content here"}`
- Response: `{"prediction": "REAL"}` or `{"prediction": "FAKE"}`

### POST /scrape
Extracts text from a given URL.
- Payload: `{"url": "https://example.com/news-article"}`
- Response: `{"text": "Extracted text content..."}`

## Dataset
The model is trained on the Fake or Real News dataset, which contains several thousand labeled articles. The dataset is used to train a Naive Bayes classifier after extensive text preprocessing including tokenization, stopword removal, and lemmatization.

## License
MIT License
