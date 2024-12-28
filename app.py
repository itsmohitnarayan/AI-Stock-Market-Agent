from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from textblob import TextBlob
import yfinance as yf  # For symbol resolution
import json

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Function to fetch stock symbol from Yahoo Finance
def fetch_stock_symbol(company_name):
    try:
        ticker = yf.Ticker(company_name)
        if ticker.info and "symbol" in ticker.info:
            return ticker.info["symbol"]
        return None
    except Exception as e:
        logging.error(f"Error resolving stock symbol with yfinance: {e}")
        return None

# Function to fetch stock symbol from Alpha Vantage
def fetch_stock_symbol_alpha_vantage(company_name):
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        best_match = data.get("bestMatches", [])
        if best_match:
            return best_match[0]["1. symbol"]  # Return the first matching symbol
        return None
    except requests.RequestException as e:
        logging.error(f"Error resolving stock symbol with Alpha Vantage: {e}")
        return None

# Function to fetch stock data
def get_stock_data(stock_symbol):
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("Time Series (5min)", {})
    except requests.RequestException as e:
        logging.error(f"Error fetching stock data for {stock_symbol}: {e}")
        return {}

# Function to calculate RSI
def calculate_rsi(stock_data):
    if not stock_data:
        return 0
    closes = [float(item["4. close"]) for item in stock_data.values()]
    gain, loss = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gain.append(delta)
            loss.append(0)
        else:
            gain.append(0)
            loss.append(abs(delta))
    avg_gain = sum(gain) / len(gain)
    avg_loss = sum(loss) / len(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to fetch news
def fetch_news(stock):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CX = os.getenv("SEARCH_ENGINE_ID")
    query = f"{stock} stock news"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        items = response.json().get("items", [])
        news = [item["title"] + " - " + item["snippet"] for item in items[:5]]
        return news
    except requests.RequestException as e:
        logging.error(f"Error fetching news: {e}")
        return ["Failed to fetch news"]

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        return {"compound": sentiment_score}
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return {"compound": 0}

# Function to query LLM
def query_llm(stock, rsi, sentiment, news_summary):
    prompt = f"""
    Stock: {stock}
    RSI: {rsi}
    Sentiment Score: {sentiment['compound']}
    News Summary: {news_summary}
    Provide a recommendation (Buy, Sell, or Hold) and explain your reasoning.
    """
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return {"decision": response.text, "explanation": response.text}
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return {"error": "Failed to query LLM"}

# Analyze endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    user_input = data.get("user_input", "")
    company_name_or_symbol = user_input.split()[-1]

    # Resolve stock symbol
    stock_symbol = fetch_stock_symbol(company_name_or_symbol) or \
                   fetch_stock_symbol_alpha_vantage(company_name_or_symbol) or \
                   company_name_or_symbol

    if not stock_symbol:
        return jsonify({"error": "Unable to resolve stock symbol. Please check the company name or symbol."}), 400

    stock_data = get_stock_data(stock_symbol)
    rsi = calculate_rsi(stock_data)
    news = fetch_news(stock_symbol)
    sentiment = analyze_sentiment(" ".join(news))
    news_summary = " ".join(news)
    llm_response = query_llm(stock_symbol, rsi, sentiment, news_summary)

    response = {
        "stock": stock_symbol,
        "rsi": rsi,
        "sentiment": sentiment,
        "news_summary": news_summary,
        "llm_decision": llm_response.get("decision", "No recommendation"),
        "llm_reasoning": llm_response.get("explanation", "No explanation available")
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
