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
selected_symbol = st.sidebar.text_input("Or enter IDX stock (e.g., BBCA.JK)", "BBRI.JK").strip().upper()
future_days = st.sidebar.slider("Days into the future for prediction", 1, 30, 7)

@st.cache_data(ttl=600)
def load_data(symbol):
    try:
        df = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}.")
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'].rename(columns={symbol: 'Close'})
        else:
            df = df[['Close']]
        
        df = df.reset_index()
        
        # Ensure 'Close' is numeric and a Series
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Validate data
        if df['Close'].isna().all():
            raise ValueError(f"No valid 'Close' data for {symbol}.")
        
        return df
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def add_indicators(df):
    try:
        # Ensure 'Close' is a 1D Pandas Series
        close_series = pd.Series(df['Close'].values, index=df.index)
        
        # Calculate RSI
        df['RSI'] = RSIIndicator(close=close_series).rsi()
        
        # Calculate MACD
        df['MACD'] = MACD(close=close_series).macd()
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=close_series)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        return df
    except Exception as e:
        st.error(f"Error adding indicators: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def train_model(df):
    try:
        df = df.dropna()
        if df.empty:
            raise ValueError("No valid data after dropping NaNs.")
        
        features = ['RSI', 'MACD', 'BB_High', 'BB_Low']
        if not all(col in df.columns for col in features):
            raise ValueError(f"Missing required columns: {features}")
        
        X = df[features]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return model, X_test, y_test, predictions
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

@st.cache_data(ttl=3600)
def get_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return 0.0

# Main workflow
def main():
    # Validate input
    if not selected_symbol and not user_file:
        st.warning("Please provide a stock symbol or upload a CSV file.")
        return
    
    # Load data
    df = None
    if user_file:
        try:
            df = pd.read_csv(user_file)
            if 'Date' not in df.columns or 'Close' not in df.columns:
                st.error("CSV must contain 'Date' and 'Close' columns.")
                return
            df['Date'] = pd.to_datetime(df['Date'])
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            if len(df) < 50:
                st.error("CSV must contain at least 50 data points.")
                return
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return
    else:
        df = load_data(selected_symbol)
    
    if df is None:
        return
    
    # Add indicators
    df = add_indicators(df)
    if df is None:
        return
    
    # Train model
    model, X_test, y_test, predictions = train_model(df)
    if model is None:
        return
    
    # Future prediction
    try:
        future_date = df['Date'].iloc[-1] + timedelta(days=future_days)
        future_features = df[['RSI', 'MACD', 'BB_High', 'BB_Low']].iloc[[-1]].values
        future_price_pred = model.predict(future_features)
        future_price_val = float(future_price_pred[0])
        last_close_val = float(df['Close'].iloc[-1])
        trend_arrow = "📉" if future_price_val < last_close_val else "📈"
        
        st.success(f"{trend_arrow} Predicted Close Price on {future_date.date()}: {future_price_val:.2f}")
    except Exception as e:
        st.error(f"Error predicting future price: {str(e)}")
        return
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], mode='lines', name='BB High', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], mode='lines', name='BB Low', line=dict(dash='dot')))
    fig.update_layout(title=f"{selected_symbol} Stock Price and Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment (placeholder, needs real news data)
    news_text = f"Stock news for {selected_symbol}"  # Replace with actual news data
    sentiment_score = get_sentiment(news_text)
    st.write(f"📰 Sentiment score: {sentiment_score:.2f} (Positive > 0, Negative < 0)")
    
    # Predictions table
    result_df = pd.DataFrame({
        'Date': df['Date'].iloc[-len(y_test):],
        'Actual': y_test.values,
        'Predicted': predictions
    })
    st.subheader("Actual vs Predicted Prices")
    st.dataframe(result_df)

if __name__ == "__main__":
    main()
