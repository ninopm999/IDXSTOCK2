# IDX High Dividend 20 Stock Price Predictor with Technical Indicators & Sentiment

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import yfinance as yf
import plotly.graph_objects as go
import requests
from textblob import TextBlob
from functools import lru_cache

st.set_page_config(page_title="IDX Stock Predictor + Indicators + Sentiment", layout="wide")
st.title("📈 IDX High Dividend 20 – Smart Stock Predictor")

# --- Sidebar Inputs ---
st.sidebar.header("Upload or Select Your Stock Data")
user_file = st.sidebar.file_uploader("Upload CSV with Date, Open, High, Low, Close, Volume", type=['csv'])
selected_symbol = st.sidebar.text_input("Or enter IDX stock symbol (e.g. BBCA.JK)", value="ADRO.JK")

# --- Data Load Function ---
def load_data(symbol):
    data = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    return data

# --- Feature Engineering ---
def add_indicators(data):
    close_series = data['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    elif isinstance(close_series.values, np.ndarray) and close_series.values.ndim == 2:
        close_series = close_series.values.squeeze()
    data['RSI'] = RSIIndicator(close=close_series).rsi()
    data['MACD'] = MACD(close=close_series).macd()
    bb = BollingerBands(close=close_series)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data.dropna()

# --- Train Model ---
def train_model(data):
    features = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Day', 'Month', 'Year']
    X = data[features]
    y = data[['Close']].values.squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return model, r2, mae, X_test, y_test, predictions

# --- News API Fetch + Caching ---
@st.cache_data(ttl=3600)
def fetch_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}+stock&sortBy=publishedAt&language=en&apiKey={api_key}"
    response = requests.get(url)
    return response.json().get("articles", [])

# --- Load & Process Data ---
if user_file:
    df = pd.read_csv(user_file)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    df = load_data(selected_symbol)
    st.info(f"Loaded data for: {selected_symbol}")

st.write("✅ Data Preview:", df.tail())
df = add_indicators(df)

# --- Train & Display Results ---
model, r2, mae, X_test, y_test, predictions = train_model(df)
st.success("Model trained successfully!")
st.metric("Model R² Accuracy", f"{r2*100:.2f}%")
st.metric("Mean Absolute Error", f"{mae:.2f} IDR")

# --- Chart Actual vs Predicted ---
with st.expander("📊 View Actual vs Predicted Close Price"):
    fig = go.Figure()
    test_dates = df['Date'].iloc[-len(y_test):]
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions, mode='lines', name='Predicted'))
    fig.update_layout(title="Actual vs Predicted Close Price", xaxis_title="Date", yaxis_title="Price (IDR)")
    st.plotly_chart(fig, use_container_width=True)

    # Export results as CSV
    result_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test,
        'Predicted': predictions
    })
    csv_export = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions as CSV", data=csv_export, file_name="predicted_vs_actual.csv", mime="text/csv")

# --- Predict Future ---
st.subheader("📅 Predict Future Price")
days_ahead = st.slider("Days into the future", 1, 30, 5)
future_date = df['Date'].iloc[-1] + timedelta(days=days_ahead)
last_row = df.iloc[-1]
future_features = pd.DataFrame([{
    'Open': last_row['Open'],
    'High': last_row['High'],
    'Low': last_row['Low'],
    'Volume': last_row['Volume'],
    'RSI': last_row['RSI'],
    'MACD': last_row['MACD'],
    'BB_High': last_row['BB_High'],
    'BB_Low': last_row['BB_Low'],
    'Day': future_date.day,
    'Month': future_date.month,
    'Year': future_date.year
}])

future_price = float(model.predict(future_features)[0])
last_close = float(df['Close'].iloc[-1])
trend_arrow = "📉" if future_price < last_close else "📈"

st.success(f"{trend_arrow} Predicted Close Price on {future_date.date()}: Rp {future_price:,.2f}")

# --- Technical Indicators Overview ---
with st.expander("📉 Technical Indicators Today"):
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}", delta=None)
    col2.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}", delta=None)
    col3.metric("BB Width", f"{(df['BB_High'].iloc[-1] - df['BB_Low'].iloc[-1]):.2f}")

# --- News Sentiment Tab ---
with st.expander("📰 Latest News & Sentiment (Beta)"):
    try:
        news_api_key = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else ""
        if news_api_key:
            query = selected_symbol.replace(".JK", "")
            articles = fetch_news(query, news_api_key)[:5]
            if articles:
                for article in articles:
                    st.markdown(f"**{article['title']}**")
                    st.markdown(f"*{article['source']['name']} - {article['publishedAt'][:10]}*")
                    st.markdown(f"[{article['url']}]({article['url']})")
                    # Sentiment
                    sentiment = TextBlob(article['title']).sentiment.polarity
                    emoji = "👍 Positive" if sentiment > 0 else ("😐 Neutral" if sentiment == 0 else "👎 Negative")
                    st.markdown(f"**Sentiment:** {emoji}")
                    st.markdown("---")
            else:
                st.info("No recent news found for this stock.")
        else:
            st.warning("🔐 NewsAPI key not set. Add it via Streamlit secrets for live headlines.")
    except Exception as e:
        st.error(f"❌ Failed to load news: {e}")
