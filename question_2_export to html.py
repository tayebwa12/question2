import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# 1. Load your cleaned FAO dataset (adjust filename if needed)
df = pd.read_csv('./clean_fao_food_price_index.csv')

# Ensure correct column names
# Suppose your dataset has columns: Date, Food Price Index, Meat, Dairy, Cereals, Oils, Sugar
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Food Price Index']].rename(columns={'Date': 'ds', 'Food Price Index': 'y'})

# 2. Fit a Prophet model
model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model.fit(df)

# 3. Create future dataframe for forecasting
future = model.make_future_dataframe(periods=12, freq='M')  # forecast 12 months ahead
forecast = model.predict(future)

# 4. Build a Plotly figure with confidence intervals
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper CI', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower CI', line=dict(dash='dot')))

fig.update_layout(
    title='FAO Food Price Index Forecast',
    xaxis_title='Date',
    yaxis_title='Food Price Index',
    template='plotly_white'
)

# 5. Export to HTML
fig.write_html('prophet_forecast.html')
print("✅ Exported to prophet_forecast.html")
