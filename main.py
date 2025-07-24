import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Download stock data
df = yf.download('AAPL', start='2018-01-01', end='2023-12-31', auto_adjust=True)[['Close']].dropna()

# 1. Historical Plot
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Historical Prices')
plt.title('AAPL Stock Closing Price (2018–2023)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("1_historical_plot.png")
plt.close()

# 2. ARIMA Forecast
model_arima = ARIMA(df['Close'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast_arima = model_arima_fit.forecast(steps=30)
forecast_dates = pd.date_range(df.index[-1], periods=30, freq='B')

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Historical')
plt.plot(forecast_dates, forecast_arima, label='ARIMA Forecast', color='orange')
plt.title('ARIMA Forecast for AAPL')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("2_arima_forecast.png")
plt.close()

# 3. Prophet Forecast
df_prophet = df.reset_index()
df_prophet = df_prophet[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet['y'] = pd.to_numeric(df_prophet['y'])  # fix for TypeError

model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)
fig_prophet = model_prophet.plot(forecast_prophet)
fig_prophet.savefig("3_prophet_forecast.png")

# 4. LSTM Forecast
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y, epochs=2, batch_size=32, verbose=1)

predicted = model_lstm.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

plt.figure(figsize=(14, 6))
plt.plot(df.index[look_back:], df['Close'].values[look_back:], label='Actual')
plt.plot(df.index[look_back:], predicted_prices, label='LSTM Prediction', color='red')
plt.title('LSTM Prediction vs Actual (AAPL)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("4_lstm_forecast.png")
plt.close()
