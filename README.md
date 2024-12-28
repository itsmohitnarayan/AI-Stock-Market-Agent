# AI-Stock-Market-Agent

An agent for stock market analysis using AI. This application provides insights on stocks based on real-time market data, news sentiment, and advanced technical analysis.

## Features

- Fetches real-time stock data and news.
- Analyzes stock performance using indicators like RSI.
- Interprets sentiment from financial news.
- Provides actionable insights and recommendations (Buy, Sell, Hold) using Google Gemini LLM.
- Displays current stock prices in the sidebar.

## Preview

![Screenshot (30)](https://github.com/user-attachments/assets/05879936-5970-4be7-9fdd-6f42a48832ba)
![Screenshot (31)](https://github.com/user-attachments/assets/f89ab89f-28b8-435b-827f-73d09a0bf9f0)
![Screenshot (32)](https://github.com/user-attachments/assets/146962f4-c3e2-4ea4-9838-8356bdacc3b0)
![Screenshot (33)](https://github.com/user-attachments/assets/087516ae-0886-4089-ba7e-5ac639d29016)

## Setup

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/AI-Stock-Market-Agent.git
   cd AI-Stock-Market-Agent

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages**:
```
pip install -r requirements.txt
```


4. **Create a .env file with the necessary API keys**:
```
ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key"
GOOGLE_API_KEY="your_google_api_key"
SEARCH_ENGINE_ID="your_search_engine_id"
GEMINI_API_KEY="your_gemini_api_key"
```

**Running the Application**
*Run the Flask backend:*
```
python app.py
```

*Run the Streamlit app:*
```
streamlit run streamlit_app.py
```

## Usage
1.Open the Streamlit app in your browser. The default URL is ```http://localhost:8501```.
2.View current stock prices in the sidebar.
3.Enter a company name or stock symbol in the input field and click "Analyze Stock".
4.View the analysis results, including RSI, sentiment score, news summary, and LLM decision.

## Testing
1.*Run the unit tests*:
```
python -m unittest test_app.py
```

## File Structure
- ``app.py``: Flask backend for stock analysis & Streamlit frontend for user interaction.
- ``test_app.py``: Unit tests for the Flask backend.
- ``.env``: Environment variables (not included in the repository).
- ``.gitignore``: Specifies files and directories to be ignored by Git.
- ``README.md``: Project documentation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
