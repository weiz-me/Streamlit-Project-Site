import streamlit as st
import time
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os

# Set Streamlit page config
st.set_page_config(page_icon="🧑‍💻", layout="wide")
st.title("📈 RNN Stock Prediction (10 Years Historical Data)")

# Ensure directory for models exists
if not os.path.exists("stock"):
    os.makedirs("stock")

# Initialize session state for dataframe
if "df1" not in st.session_state:
    st.session_state.df1 = pd.DataFrame(columns=[
        "Ticker", "Short Name", "Real Stock Price", "Predicted Stock Price", "Decision", "Percent Change"
    ])

# Define stock tickers
tickers = [
    'SPY', 'WMT', 'AMZN', 'AAPL', 'CVS', 'UNH', 'TSLA', 'META', 'VZ', 'T',
    'COST', 'PG', 'HD', 'JPM', 'DIS', 'BAC', 'XOM', 'GOOGL', 'GM', 'MA',
    'MSFT', 'CMCSA', 'CVX', 'CSCO', 'ABT', 'KO', 'PEP', 'NKE', 'PYPL', 'V',
    'INTC', 'JNJ', 'BLK', 'NEE', 'CMC', 'UPS', 'LOW', 'MCD', 'MET'
]

# Graph toggle and inputs
month = st.number_input("📅 Enter Start Month (1-12)", min_value=1, max_value=12, value=7)
show_graph = st.selectbox("📊 Show Graph?", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Stock model class
class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.info = yf.Ticker(ticker)
        try:
            self.short_name = self.info.info.get("shortName", "Unknown")
        except Exception:
            self.short_name = "Unknown"

    def train(self, period="5y"):
        hist = self.info.history(period=period)
        if hist.empty:
            raise ValueError("Empty history data.")

        self.training_set = hist.iloc[:, :1].values
        sc = MinMaxScaler(feature_range=(0, 1))
        training_scaled = sc.fit_transform(self.training_set)

        X_train, y_train = [], []
        for i in range(120, len(training_scaled)):
            X_train.append(training_scaled[i - 120:i, 0])
            y_train.append(training_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        for _ in range(5):
            model.add(LSTM(60, return_sequences=True))
            model.add(Dropout(0.2))
        model.add(LSTM(60))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        self.model = model
        self.scaler = sc

    def save(self):
        self.model.save(f"stock/stock_{self.ticker}.keras")
        np.save(f"stock/traindata_{self.ticker}.npy", self.training_set)

    def load(self):
        try:
            self.model = load_model(f"stock/stock_{self.ticker}.keras")
            self.training_set = np.load(f"stock/traindata_{self.ticker}.npy")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit_transform(self.training_set)
        except Exception as e:
            raise FileNotFoundError(f"Model or training data not found for {self.ticker}: {e}")

    def predict_next(self):
        hist = self.info.history(period="1y")
        real = hist.iloc[:, :1].values
        if len(real) < 150:
            raise ValueError("Not enough data to predict.")

        inputs = self.scaler.transform(real[-150:])
        X_test = []
        for i in range(120, len(inputs) + 1):
            X_test.append(inputs[i - 120:i, 0])
        X_test = np.array(X_test).reshape((-1, 120, 1))
        pred = self.model.predict(X_test, verbose=0)
        return real, self.scaler.inverse_transform(pred)

    def plot_prediction(self, month):
        start = datetime.datetime(datetime.datetime.now().year, month, 1)
        end = datetime.datetime.now()

        real, predicted = self.predict_next()
        hist = self.info.history(start=start, end=end)
        real_display = hist.iloc[:, :1].values
        dates = hist.index

        plt.figure(figsize=(10, 4))
        plt.plot(dates, real_display[-len(dates):], color="red", label="Real Price")
        plt.plot(dates, predicted[-len(dates)-1:-1], color="blue", label="Predicted")
        plt.plot(dates, [predicted[-1]] * len(dates), linestyle="--", color="green", label="Next Day Forecast")
        plt.title(f"{self.ticker} Price Prediction")
        plt.xticks(rotation=45)
        plt.legend()
        return plt, real_display, predicted

# Show stock prediction
def show_stock(ticker, month, graph_flag):
    stock = Stock(ticker)
    try:
        stock.load()
        plt_obj, real, pred = stock.plot_prediction(month)

        real_price = float(real[-1])
        predicted_price = float(pred[-1])
        decision = "BUY" if predicted_price > real_price else "SELL"
        percent_change = round((predicted_price - real_price) / real_price * 100, 2)

        st.markdown(f"### {ticker} - {stock.short_name}")
        st.write(f"**Latest Price**: {real_price:.2f}")
        st.write(f"**Predicted Next-Day**: {predicted_price:.2f}")
        st.success(f"### Decision: {decision}")

        if graph_flag:
            st.pyplot(plt_obj)

        new_row = {
            "Ticker": ticker,
            "Short Name": stock.short_name,
            "Real Stock Price": real_price,
            "Predicted Stock Price": predicted_price,
            "Decision": decision,
            "Percent Change": percent_change,
        }
        st.session_state.df1 = pd.concat([st.session_state.df1, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")

# Retrain stock
def retrain_stock(ticker):
    try:
        st.write(f"🔄 Training {ticker}...")
        stock = Stock(ticker)
        stock.train()
        stock.save()
        st.success(f"✅ Trained and saved {ticker}")
    except Exception as e:
        st.error(f"❌ Failed training {ticker}: {e}")

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📈 Predict All"):
        for t in tickers:
            show_stock(t, month, show_graph)

with col2:
    if st.button("🛠️ Retrain All"):
        for t in tickers:
            if t not in ["SPY", "WMT"]:  # example skip list
                retrain_stock(t)

# Show table
if not st.session_state.df1.empty:
    df_sorted = st.session_state.df1.sort_values("Percent Change", ascending=False)
    st.markdown("### 📋 Summary Table (Sorted by % Change)")
    st.dataframe(df_sorted.reset_index(drop=True))
