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
df = yf.download('AAPL',start='2018-01-01', end='2023-12-31', auto_adjust=True)[['Close']].dropna()
# Ensure datetime index and frequency
df.index = pd.to_datetime(df.index)
df = df.asfreq('B')  # Business day frequency

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
plt.show()
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
plt.show()
plt.close()

# 3. Prophet Forecast
df_prophet = df.reset_index()

# Check if 'Date' and 'Close' columns exist
if 'Date' not in df_prophet.columns or 'Close' not in df_prophet.columns:
    print("Missing 'Date' or 'Close' column in DataFrame.")
    print(df_prophet.columns)
else:
    df_prophet = df_prophet[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    print("Column types:\n", df_prophet.dtypes)
    print("\nPreview of data:\n", df_prophet.head())
    # Check the contents
    print(df_prophet.head())

    print("df_prophet shape:", df_prophet.shape)
    print("df_prophet columns:", df_prophet.columns)
    print(df_prophet.head())

    df_prophet.columns = ['ds', 'y']
    # Convert y column safely to numeric
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')  # convert invalids to NaN
    df_prophet.dropna(subset=['y'], inplace=True)  # remove rows where y is NaN

    # Fit and Forecast
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast_prophet = model_prophet.predict(future)

    # Plot and save
    fig_prophet = model_prophet.plot(forecast_prophet)
    fig_prophet.savefig("3_prophet_forecast.png")
    fig_prophet.show()  # Only if you're in Jupyter; otherwise do this:
    plt.show()


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
plt.show()
plt.close()



import matplotlib.dates as mdates

# Create a combined figure with 4 subplots
fig, axs = plt.subplots(4, 1, figsize=(14, 20))  # 4 rows, 1 column

# 1. Historical Plot
axs[0].plot(df.index, df['Close'], label='Historical Prices', color='blue')
axs[0].set_title('AAPL Historical Prices')
axs[0].set_ylabel('Price (USD)')
axs[0].grid(True)
axs[0].legend()

# 2. ARIMA Forecast
axs[1].plot(df.index, df['Close'], label='Historical', color='blue')
axs[1].plot(forecast_dates, forecast_arima, label='ARIMA Forecast', color='orange')
axs[1].set_title('ARIMA Forecast')
axs[1].set_ylabel('Price (USD)')
axs[1].grid(True)
axs[1].legend() 

# 3. Prophet Forecast
axs[2].plot(df_prophet['ds'], df_prophet['y'], label='Historical', color='blue')
axs[2].plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet Forecast', color='green')
axs[2].set_title('Prophet Forecast')
axs[2].set_ylabel('Price (USD)')
axs[2].grid(True)
axs[2].legend()

# 4. LSTM Prediction
axs[3].plot(df.index[look_back:], df['Close'].values[look_back:], label='Actual', color='blue')
axs[3].plot(df.index[look_back:], predicted_prices, label='LSTM Prediction', color='red')
axs[3].set_title('LSTM Prediction')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Price (USD)')
axs[3].grid(True)
axs[3].legend()

# Format x-axis for better readability
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("combined_forecast.png")
plt.show()
