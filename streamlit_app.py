import streamlit as st
import requests
import yfinance as yf
import os

# Streamlit app configuration
st.set_page_config(page_title="AI Stock Market Agent", layout="wide")

# Title and description
st.title("AI-Powered Stock Market Analyst")
st.markdown(
    """
    An agent for stock market analysis using AI. This application provides insights on stocks based on real-time market data, news sentiment, and advanced technical analysis.
    This app uses cutting-edge AI to provide insights on stocks.
    """
)

# Sidebar for current stock prices and link to stock symbol info
st.sidebar.title("Stock Prices")
st.sidebar.markdown("### Current Prices")

# Function to fetch current stock price
def fetch_current_price(stock_symbol):
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_symbol}&apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return float(data["Global Quote"]["05. price"])
    except Exception as e:
        st.sidebar.error(f"Error fetching current price for {stock_symbol}: {e}")
        return None

# List of stock symbols to display in the sidebar
stock_symbols = ["TSLA", "AAPL", "GOOGL", "AMZN", "MSFT"]

# Display current prices in the sidebar
for symbol in stock_symbols:
    price = fetch_current_price(symbol)
    if price:
        st.sidebar.write(f"{symbol}: ${price}")

# Link to get all stock symbol info
st.sidebar.markdown("[Get all stock symbol info](https://stockanalysis.com/stocks/)")

# Input section for the user to specify the company name or stock symbol
user_input = st.text_input("Enter Company Name or Stock Symbol (e.g., Tesla, TSLA, SUZLON):", "")

# Helper function to get the stock symbol using yfinance
def fetch_stock_symbol(company_name):
    try:
        ticker = yf.Ticker(company_name)
        info = ticker.info
        if "symbol" in info:
            return info["symbol"]
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching stock symbol: {e}")
        return None

# Analyze button
if st.button("Analyze Stock"):
    if user_input.strip():
        st.info(f"Fetching analysis for: {user_input.title()}")

        # Check if the input is a company name or stock symbol
        stock_symbol = fetch_stock_symbol(user_input)
        if stock_symbol:
            st.success(f"Identified Stock Symbol: {stock_symbol}")
        else:
            st.error("Could not find a stock symbol for the provided company name or symbol.")
            stock_symbol = None

        if stock_symbol:
            # Call the Flask backend
            api_url = "http://127.0.0.1:5000/analyze"  # Ensure this matches the backend endpoint
            payload = {"user_input": f"Analyze {stock_symbol}"}

            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Display results
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.subheader(f"Results for {data['stock']}")
                    st.write(f"**RSI:** {data['rsi']:.2f}")
                    st.write(f"**Sentiment Score:** {data['sentiment']['compound']:.2f}")
                    st.write(f"**News Summary:** {data['news_summary']}")
                    st.write(f"**LLM Decision:** {data['llm_decision']}")
                    st.write(f"**LLM Reasoning:** {data['llm_reasoning']}")

            except requests.RequestException as e:
                st.error(f"Error fetching data: {e}")