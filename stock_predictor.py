import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from textblob import TextBlob
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IDX Stock Predictor")
st.title("📈 IDX High Dividend 20 – Smart Stock Predictor")

# Sidebar inputs
st.sidebar.markdown("### 🛠 Options")
user_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
selected_symbol = st.sidebar.text_input("Or enter IDX stock (e.g. BBCA.JK)", "BBRI.JK")
future_days = st.sidebar.slider("Days into the future for prediction", 1, 30, 7)

@st.cache_data(ttl=600)
def load_data(symbol):
    df = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=False)
    df.reset_index(inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].squeeze()
    return df

def add_indicators(df):
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()
    elif isinstance(close_series.values, np.ndarray) and close_series.values.ndim == 2:
        close_series = pd.Series(close_series.values.flatten(), index=close_series.index)
    
    df['RSI'] = RSIIndicator(close=close_series).rsi()
    df['MACD'] = MACD(close=close_series).macd()
    bb = BollingerBands(close=close_series)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

@st.cache_data(ttl=1800)
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

@st.cache_data(ttl=3600)
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Load data
df = pd.read_csv(user_file) if user_file else load_data(selected_symbol)
df['Date'] = pd.to_datetime(df['Date'])
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

# Sentiment
news_text = f"Stock news for {selected_symbol}"
sentiment_score = get_sentiment(news_text)
st.write(f"📰 Sentiment score: {sentiment_score:.2f}")

# Predictions table
result_df = pd.DataFrame({
    'Date': df['Date'].iloc[-len(y_test):],
    'Actual': y_test.values,
    'Predicted': predictions
})
st.dataframe(result_df)
