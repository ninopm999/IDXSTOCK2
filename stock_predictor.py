# IDX High Dividend 20 Stock Price Predictor with Enhanced UI/UX

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
import plotly.express as px
import requests
from textblob import TextBlob
from functools import lru_cache

st.set_page_config(page_title="📊 IDX Stock Predictor", layout="wide")
st.title("💹 IDX High Dividend 20 – Smart Stock Predictor")

# Sidebar Summary Card
st.sidebar.markdown("### 🛠 Options")
user_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
selected_symbol = st.sidebar.text_input("Or enter IDX stock (e.g. BBCA.JK)", value="ADRO.JK")

# Load data
@st.cache_data(ttl=600)
def load_data(symbol):
    df = yf.download(symbol, start="2024-01-01", end=datetime.today().strftime('%Y-%m-%d"))
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=1800)
def train_model(df):
    X = df[['Open','High','Low','Volume','RSI','MACD','BB_High','BB_Low','Day','Month','Year']]
    y = df[['Close']].values.squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_test, y_test, preds

@st.cache_data(ttl=3600)
def fetch_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}+stock&sortBy=publishedAt&language=en&apiKey={api_key}"
    r = requests.get(url)
    return r.json().get("articles", [])

# Feature engineering
@st.cache_data(ttl=3600)
def add_indicators(df):
    df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    df['MACD'] = MACD(close=df['Close']).macd()
    bb = BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df.dropna()

# Load
df = pd.read_csv(user_file) if user_file else load_data(selected_symbol)
df['Date'] = pd.to_datetime(df['Date'])
df = add_indicators(df)
model, X_test, y_test, predictions = train_model(df)
test_dates = df['Date'].iloc[-len(y_test):]

# Tabs for layout
tabs = st.tabs(["🏦 Stock Data", "📈 Model Prediction", "📊 Indicators", "📰 Sentiment"])

with tabs[0]:
    st.subheader(f"Stock Preview: {selected_symbol}")
    st.dataframe(df.tail(), use_container_width=True)

with tabs[1]:
    st.subheader("📈 Actual vs Predicted Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test, name="Actual"))
    fig.add_trace(go.Scatter(x=test_dates, y=predictions, name="Predicted"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (IDR)")
    st.plotly_chart(fig, use_container_width=True)
    result_df = pd.DataFrame({'Date': test_dates, 'Actual': y_test, 'Predicted': predictions})
    st.download_button("📥 Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv")

    st.subheader("🧠 Model Performance")
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    st.metric("R² Accuracy", f"{r2*100:.2f}%")
    st.metric("MAE", f"{mae:.2f} IDR")

    future_date = df['Date'].iloc[-1] + timedelta(days=5)
    last_row = df.iloc[-1]
    future_features = pd.DataFrame([{ 'Open': last_row['Open'], 'High': last_row['High'], 'Low': last_row['Low'], 'Volume': last_row['Volume'], 'RSI': last_row['RSI'], 'MACD': last_row['MACD'], 'BB_High': last_row['BB_High'], 'BB_Low': last_row['BB_Low'], 'Day': future_date.day, 'Month': future_date.month, 'Year': future_date.year }])
    future_price = float(model.predict(future_features)[0])
    trend = "📈" if future_price > last_row['Close'] else "📉"
    st.success(f"{trend} Future Close Price ({future_date.date()}): Rp {future_price:,.2f}")

with tabs[2]:
    st.subheader("📊 Technical Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
    col2.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
    col3.metric("BB Width", f"{df['BB_High'].iloc[-1] - df['BB_Low'].iloc[-1]:.2f}")

with tabs[3]:
    st.subheader("📰 News & Sentiment")
    try:
        key = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else ""
        if key:
            q = selected_symbol.replace(".JK", "")
            articles = fetch_news(q, key)[:10]
            sentiments, news_data = [], []
            for a in articles:
                title = a['title']
                score = TextBlob(title).sentiment.polarity
                tag = "Positive" if score > 0 else "Neutral" if score == 0 else "Negative"
                emoji = "👍" if tag == "Positive" else "😐" if tag == "Neutral" else "👎"
                st.markdown(f"**{title}**\n- {emoji} {tag}")
                sentiments.append(tag)
                news_data.append({"Title": title, "Sentiment": tag})
            pie = px.pie(pd.DataFrame(sentiments, columns=["Sentiment"]).value_counts().reset_index(name="Count"), names="Sentiment", values="Count", title="Sentiment Overview")
            st.plotly_chart(pie, use_container_width=True)
            st.download_button("📰 Download Sentiment", pd.DataFrame(news_data).to_csv(index=False), file_name="news_sentiment.csv")
        else:
            st.warning("🔐 Add your NewsAPI key in Streamlit secrets")
    except Exception as e:
        st.error(f"❌ News error: {e}")
