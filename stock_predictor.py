import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Function to fetch and cache stock data
@st.cache_data
def load_data(symbol, start="2010-01-01", end="2025-06-16"):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            return None
        return data
    except Exception:
        return None

# Function to preprocess data and create features
def preprocess_data(data, look_back=30, horizon=1):
    data = data.copy()
    # Create lagged price features
    for i in range(1, look_back + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    # Target is the price 'horizon' days ahead
    data['target'] = data['Close'].shift(-horizon)
    # Remove rows with NaN values
    data = data.dropna()
    X = data[[f'lag_{i}' for i in range(1, look_back + 1)]]
    y = data['target']
    return X, y, data

# Streamlit app layout
st.title("Indonesia Stock Price Predictor")
st.markdown("""
    This app predicts stock prices on the Indonesia Stock Exchange (IDX).
    Select a stock and prediction horizon below.
""")

# Stock selection
stocks = {
    'BBCA.JK': 'Bank Central Asia',
    'TLKM.JK': 'Telekomunikasi Indonesia',
    'BMRI.JK': 'Bank Mandiri'
}
selected_stock = st.selectbox("Select a Stock", list(stocks.keys()), format_func=lambda x: stocks[x])

# Prediction horizon
horizon = st.slider("Prediction Horizon (Trading Days)", 1, 30, 1)

# Button to trigger prediction
if st.button("Predict"):
    # Load data
    data = load_data(selected_stock)
    if data is None:
        st.error("Unable to fetch data for this stock. Please try another.")
    else:
        # Preprocess data
        X, y, processed_data = preprocess_data(data, look_back=30, horizon=horizon)

        # Load pre-trained model (assumes model is pre-saved)
        try:
            model = joblib.load(f"{selected_stock}_model.pkl")
        except FileNotFoundError:
            st.error("Pre-trained model not found. Please ensure models are trained and saved.")
            st.stop()

        # Prepare the latest features for prediction
        latest_features = X.iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(latest_features)[0]

        # Display prediction
        st.subheader("Prediction")
        st.write(f"Predicted price for {stocks[selected_stock]} in {horizon} trading days: **{predicted_price:.2f} IDR**")

        # Visualization
        st.subheader("Historical Prices and Prediction")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Close'], label='Historical Prices', color='blue')
        last_date = data.index[-1]
        future_date = last_date + pd.offsets.BDay(horizon)
        ax.plot([last_date, future_date], [data['Close'].iloc[-1], predicted_price], 
                'ro-', label='Predicted Price', linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (IDR)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Model performance (example using last 30 days as test)
        X_train, X_test = X[:-30], X[-30:]
        y_train, y_test = y[:-30], y[-30:]
        model.fit(X_train, y_train)  # Retrain for evaluation
        test_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        st.write(f"Model RMSE (based on last 30 days): **{rmse:.2f} IDR**")

# Disclaimer
st.markdown("""
    **Disclaimer**: Stock price prediction is inherently uncertain due to market volatility and external factors. 
    This app is for educational purposes only and should not be relied upon for investment decisions.
""")