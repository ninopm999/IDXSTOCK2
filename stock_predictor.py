# Fixed version of the stock predictor script
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import yfinance as yf
import plotly.graph_objects as go
import requests
from textblob import TextBlob
from functools import lru_cache
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Set page configuration
st.set_page_config(page_title="IDX Stock Predictor + Indicators + Sentiment")
st.title("📈 IDX High Dividend 20 – Smart Stock Predictor")

# Download data
symbol = "BBRI.JK"  # Example default symbol
df = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=False)

# Add technical indicators
def add_indicators(df):
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

df = add_indicators(df)

# Train model
features = ['RSI', 'MACD', 'BB_High', 'BB_Low']
df = df.dropna()
X = df[features]
y = df['Close']
model = RandomForestRegressor()
model.fit(X, y)

# Predict future price with dummy input (for example purpose)
future_features = X.tail(1)
future_price = model.predict(future_features)
last_close = df['Close'].iloc[-1]

# Fix ambiguous Series comparison and deprecated float usage
future_price_value = float(future_price[0]) if isinstance(future_price, np.ndarray) else float(future_price.iloc[0])
trend_arrow = "📉" if future_price_value < float(last_close) else "📈"

# Show result
st.success(f"{trend_arrow} Predicted Close Price: Rp{future_price_value:,.2f}")
