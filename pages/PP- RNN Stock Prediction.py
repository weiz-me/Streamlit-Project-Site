import streamlit as st
#from predict_page1 import show_predict_page
#from explore_page1 import show_explore_page


#page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
st.set_page_config(
    page_icon="🧑‍💻"
)


import streamlit as st
import time
import datetime
import pandas as pd
import yfinance as yahooFinance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras

st.title("RNN Stock Prediction")
st.write("""### RNN Stock Prediction (Past 10yrs data)""")
data = {
    'Ticker': [],
    'Short Name': [],
    'Real Stock Price': [],
    'Predicted Stock Price': [],
    'Decision': []
}      
df1 = pd.DataFrame(data)
# df1.set_index('Ticker', inplace=True)

global table1
table1 = st.dataframe(df1)

class Stock:
    def __init__(self,ticker):
        self.ticker = ticker
        self.Stockinfo = yahooFinance.Ticker(ticker)

    def train(self,dur):
        dataset_train = self.Stockinfo.history(period=dur)
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set = dataset_train.iloc[:, :1].values
        training_set_scaled = sc.fit_transform(training_set)
        X_train = []
        y_train = []
        for i in range(120, training_set_scaled.shape[0]):
            X_train.append(training_set_scaled[i-120:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        regressor = Sequential()
        regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 60))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        self.regressor = regressor
        self.sc = sc
        self.training_set = training_set
    
        
    def save(self):
        self.regressor.save(f"stock/stock_{self.ticker}.keras")
        np.save(f"stock/traindata_{self.ticker}",self.training_set)
        
    def load(self):
        self.regressor = keras.models.load_model(f"stock/stock_{self.ticker}.keras")
        self.training_set = np.load(f"stock/traindata_{self.ticker}.npy")
        self.sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = self.sc.fit_transform(self.training_set)

        
    def pred(self):
        
        dataset_test = self.Stockinfo.history(period="1y")
        real_stock_price = dataset_test.iloc[:, :1].values
        inputs = real_stock_price[-150:]
        inputs = inputs.reshape(-1,1)
        inputs = self.sc.transform(inputs)

        X_test = []
        for i in range(120, inputs.shape[0]+1):
            X_test.append(inputs[i-120:i, 0])
        X_test = np.array(X_test)
        print(X_test.shape)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = self.regressor.predict(X_test)
        predicted_stock_price = self.sc.inverse_transform(predicted_stock_price)
        return predicted_stock_price
        
    def display(self,month):
        startDate = datetime.datetime(2024, month, 1)
        endDate = datetime.datetime.now()
        predicted_stock_price = self.pred()
        dataset_display = self.Stockinfo.history(start=startDate,end=endDate)
        real_stock_price = dataset_display.iloc[:, :1].values
        ns = len(dataset_display.index)
        plt.plot(dataset_display.index, real_stock_price[-ns:], color = 'red', label = f'Real {self.ticker} Stock Price')
        plt.plot(dataset_display.index, predicted_stock_price[-ns-1:-1], color = 'blue', label = f'Predicted {self.ticker} Stock Price')
        plt.plot(dataset_display.index, [predicted_stock_price[-1] for _ in range(ns)], color = 'green', label = 'Nextday Prediction',linestyle = "--")
        plt.title(f'{self.ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{self.ticker} Stock Price')
        plt.xticks(rotation=90)

        plt.legend()
        # plt.show()

        return plt,dataset_display,real_stock_price, predicted_stock_price
    
plts = None

def show_stock(ticker,month):
    info = yahooFinance.Ticker(ticker)
    st.write(f"""### {ticker} - {info.info['shortName']}""")
    curstock = Stock(ticker)
    curstock.load()
    global plts
    global plt
    if plts:
        plt.clf()
    plt,dataset_display,real_stock_price, predicted_stock_price = curstock.display(month)
    plts = True
    st.write(f"""##### {dataset_display.index[-1]}: {real_stock_price[-1]}""")
    st.write(f"""##### Tomorrow Prediciton: {predicted_stock_price[-1]}""")
    if predicted_stock_price[-1] > real_stock_price[-1]:
        dec = "BUY"
        st.write("##### BUY")
    else:
        dec = "SELL"
        st.write("##### SELL")
    global graph
    if graph:
        st.pyplot(plt)
    global table
    data2 = {
    'Ticker': str(ticker),
    'Short Name': str(info.info['shortName']),
    'Real Stock Price': real_stock_price[-1],
    'Predicted Stock Price': predicted_stock_price[-1],
    'Decision': dec,
    'Percent Change':round(float((predicted_stock_price[-1]-real_stock_price[-1])/real_stock_price[-1]),2),
    }
    
    df2 = pd.DataFrame(data2)
    # df2.set_index('Ticker', inplace=True)
    global df1
    df1 = pd.concat([df1,df2])
    df1=df1.sort_values('Percent Change',ascending=False)
    global table2
    table2=st.table(df1)
    # global table1
    # table1.add_rows(df2)
def retrain_stock(ticker):
    info = yahooFinance.Ticker(ticker)
    # st.write(f"""Training {ticker} - {info.info['shortName']}""")
    print(f"""Training {ticker} - {info.info['shortName']}""")
    curstock = Stock(ticker)
    curstock.train("5y")
    curstock.save()
    print(f"""\t\t\t done {ticker}""")


tickers = ['SPY','WMT',
 'AMZN',
 'AAPL',
 'CVS',
 'UNH',
 'TSLA',
 'META',
 'VZ',
 'T',
 'COST',
 'PG',
 'HD',
 'JPM',
 'DIS',
 'BAC',
 'XOM',
 'GOOGL',
 'GM',
 'MA',
 'MSFT',
 'CMCSA',
 'CVX',
 'CSCO',
 'ABT',
 'KO',
 'PEP',
 'NKE',
 'PYPL',
 'V',
 'INTC',
 'JNJ',
 'BLK',
 'NEE',
 'CMC',
 'UPS',
 'LOW',
 'MCD',
 'MET']

month = st.number_input("Enter Start Month", value = 7)
graph = st.selectbox("Graph?",[0,1])
but = st.button("pred")
but2 = st.button("train")

if but:
    for ticker in tickers:
        show_stock(ticker,int(month))

if but2:
    for ticker in tickers:
        if ticker not in ["WMT","SPY"]:
            retrain_stock(ticker)

