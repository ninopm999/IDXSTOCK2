import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import requests
from textblob import TextBlob
from functools import lru_cache
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Streamlit setup
st.set_page_config(page_title="IDX Stock Predictor + Indicators + Sentiment")
st.title("📈 IDX High Dividend 20 – Smart Stock Predictor")

# Sidebar inputs
selected_symbol = st.sidebar.text_input("Enter IDX Symbol (e.g. BBRI.JK)", "BBRI.JK")
future_days = st.sidebar.slider("Days into the future for prediction", 1, 30, 7)

@st.cache_data(ttl=3600)
def load_data(symbol):
    df = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=False)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=3600)
def add_indicators(df):
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

@st.cache_data(ttl=3600)
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def train_model(df):
    df = df.dropna()
    features = ['RSI', 'MACD', 'BB_High', 'BB_Low']
    X = df[features]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions

# Load and process data
df = load_data(selected_symbol)
df = add_indicators(df)

# Train model
model, X_test, y_test, predictions = train_model(df)

# Future prediction
future_date = df['Date'].iloc[-1] + timedelta(days=future_days)
future_features = df[['RSI', 'MACD', 'BB_High', 'BB_Low']].iloc[[-1]].values
future_price_pred = model.predict(future_features)
future_price_val = float(future_price_pred[0])
last_close_val = float(df['Close'].iloc[-1])

trend_arrow = "📉" if future_price_val < last_close_val else "📈"

st.success(f"{trend_arrow} Predicted Close Price on {future_date.date()}: {future_price_val:.2f}")

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], mode='lines', name='BB High', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], mode='lines', name='BB Low', line=dict(dash='dot')))
st.plotly_chart(fig, use_container_width=True)

# Sentiment (optional, mock example)
news_text = f"Stock news for {selected_symbol}"
sentiment_score = get_sentiment(news_text)
st.write(f"📰 Sentiment score: {sentiment_score:.2f}")

# Display predictions table
result_df = pd.DataFrame({
    'Date': df['Date'].iloc[-len(y_test):],
    'Actual': y_test.values,
    'Predicted': predictions
})
st.dataframe(result_df)
