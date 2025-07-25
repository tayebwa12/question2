import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

df = pd.read_csv("clean_fao_food_price_index.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Food Price Index']]  # focus on main index
df = df.dropna()
df = df.sort_values('Date').reset_index(drop=True)

# ================================================
# 2. Basic EDA
# ================================================
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['Food Price Index'], label='Food Price Index')
plt.title('FAO Food Price Index over Time')
plt.xlabel('Date'); plt.ylabel('Index Value')
plt.legend(); plt.tight_layout()
plt.show()

print(df.describe())

# ================================================
# 3. Prophet Forecast
# ================================================
prophet_df = df.rename(columns={'Date':'ds','Food Price Index':'y'})

m = Prophet(interval_width=0.95)
m.fit(prophet_df)

future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# Plot Prophet forecast
fig1 = m.plot(forecast)
plt.title('Prophet Forecast with 95% Confidence Intervals')
plt.show()

# ================================================
# 4. ARIMA Forecast
# ================================================
# ARIMA needs just the values (assume monthly data)
series = df.set_index('Date')['Food Price Index']
# Build ARIMA(1,1,1) as a starting point
arima_model = ARIMA(series, order=(1,1,1))
arima_fit = arima_model.fit()

# Forecast next 12 months
forecast_steps = 12
arima_forecast = arima_fit.get_forecast(steps=forecast_steps)
arima_pred = arima_forecast.predicted_mean
arima_conf = arima_forecast.conf_int()

# Build timeline for ARIMA forecast
last_date = series.index[-1]
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# Plot ARIMA
plt.figure(figsize=(10,4))
plt.plot(series.index, series, label='Observed')
plt.plot(future_dates, arima_pred, label='ARIMA Forecast')
plt.fill_between(future_dates, arima_conf.iloc[:,0], arima_conf.iloc[:,1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast with 95% Confidence Intervals')
plt.xlabel('Date'); plt.ylabel('Index Value')
plt.legend(); plt.tight_layout()
plt.show()

# ================================================
# 5. KPIs
# ================================================
latest_value = df['Food Price Index'].iloc[-1]
mean_value = df['Food Price Index'].mean()
print(f"✅ Latest Food Price Index: {latest_value}")
print(f"✅ Average Food Price Index: {mean_value:.2f}")
