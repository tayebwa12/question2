import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# -----------------------------
# 1. LOAD AND CLEAN DATA
# -----------------------------
# Replace with your actual file name
df = pd.read_csv("fao_food_prices.csv")

# Ensure Date is parsed correctly
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Set Date as index for time series
df.set_index('Date', inplace=True)

# Focus on Food Price Index (main KPI)
ts = df['Food Price Index'].dropna()

# -----------------------------
# 2. EDA
# -----------------------------
print("Basic Info:\n", ts.describe())

# Plot time series
plt.figure(figsize=(12,6))
plt.plot(ts, label='Food Price Index')
plt.title('Food Price Index Over Time')
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend()
plt.tight_layout()
plt.savefig("eda_food_price_index.png")
plt.show()

# Seasonal decomposition (optional)
# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts, model='additive', period=12)
# result.plot()
# plt.show()

# -----------------------------
# 3. FORECASTING
# -----------------------------

# --- ARIMA MODEL ---
# Simple ARIMA(p,d,q)
model_arima = ARIMA(ts, order=(1,1,1))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.get_forecast(steps=12)
forecast_ci_arima = forecast_arima.conf_int()

# Plot ARIMA forecast
plt.figure(figsize=(12,6))
plt.plot(ts, label='Observed')
plt.plot(forecast_arima.predicted_mean.index, forecast_arima.predicted_mean, label='ARIMA Forecast', color='red')
plt.fill_between(forecast_ci_arima.index,
                 forecast_ci_arima.iloc[:,0],
                 forecast_ci_arima.iloc[:,1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast with Confidence Interval')
plt.legend()
plt.tight_layout()
plt.savefig("forecast_arima.png")
plt.show()

# --- PROPHET MODEL ---
df_prophet = ts.reset_index()
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)

future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# Plot Prophet forecast
fig1 = m.plot(forecast)
fig1.savefig("forecast_prophet.png")

fig2 = m.plot_components(forecast)
fig2.savefig("forecast_prophet_components.png")

# -----------------------------
# 4. KPIs
# -----------------------------
latest_value = ts.iloc[-1]
annual_change = (ts.iloc[-1] - ts.iloc[-13]) / ts.iloc[-13] * 100 if len(ts) > 13 else None

print("\n--- KEY METRICS ---")
print(f"Latest Food Price Index: {latest_value}")
if annual_change is not None:
    print(f"Year-over-Year Change: {annual_change:.2f}%")
else:
    print("Not enough data for YoY change.")

# -----------------------------
# 5. SAVE CLEANED DATA
# -----------------------------
ts.to_csv("clean_food_price_index.csv")
print("\nClean dataset saved as clean_food_price_index.csv")
