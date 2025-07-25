import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("clean_fao_food_price_index.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Food Price Index']].dropna()
df = df.sort_values('Date').reset_index(drop=True)

# =========================
# Fit Prophet once for all
# =========================
prophet_df = df.rename(columns={'Date':'ds','Food Price Index':'y'})
m = Prophet(interval_width=0.95)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# =========================
# Fit ARIMA once for all
# =========================
series = df.set_index('Date')['Food Price Index']
arima_model = ARIMA(series, order=(1,1,1))
arima_fit = arima_model.fit()
forecast_steps = 12
arima_forecast = arima_fit.get_forecast(steps=forecast_steps)
arima_pred = arima_forecast.predicted_mean
arima_conf = arima_forecast.conf_int()
last_date = series.index[-1]
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# =========================
# Build Dash app
# =========================
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Food Price Index Forecast Dashboard", style={'textAlign':'center'}),
    html.Label("Select Year Range:"),
    dcc.RangeSlider(
        id='year_slider',
        min=df['Date'].dt.year.min(),
        max=df['Date'].dt.year.max(),
        value=[df['Date'].dt.year.min(), df['Date'].dt.year.max()],
        marks={year: str(year) for year in range(df['Date'].dt.year.min(), df['Date'].dt.year.max()+1, 2)},
        step=1
    ),
    dcc.Graph(id='historical_graph'),
    html.H2("Prophet Forecast"),
    dcc.Graph(id='prophet_graph'),
    html.H2("ARIMA Forecast"),
    dcc.Graph(id='arima_graph')
])

@app.callback(
    Output('historical_graph', 'figure'),
    Input('year_slider', 'value')
)
def update_historical(year_range):
    start_year, end_year = year_range
    dff = df[(df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['Date'], y=dff['Food Price Index'], mode='lines', name='Food Price Index'))
    fig.update_layout(title="Historical Food Price Index", xaxis_title="Date", yaxis_title="Index")
    return fig

@app.callback(
    Output('prophet_graph', 'figure'),
    Input('year_slider', 'value')
)
def update_prophet(year_range):
    # Prophet forecast already prepared
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
        name='Upper CI', line=dict(dash='dot', color='lightgrey')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
        name='Lower CI', line=dict(dash='dot', color='lightgrey'),
        fill='tonexty', fillcolor='rgba(200,200,200,0.2)'
    ))
    fig.update_layout(title="Prophet Forecast (Next 12 Months)", xaxis_title="Date", yaxis_title="Index")
    return fig

@app.callback(
    Output('arima_graph', 'figure'),
    Input('year_slider', 'value')
)
def update_arima(year_range):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Observed'))
    fig.add_trace(go.Scatter(x=future_dates, y=arima_pred, mode='lines', name='ARIMA Forecast'))
    fig.add_trace(go.Scatter(
        x=future_dates, y=arima_conf.iloc[:,0], mode='lines',
        name='Lower CI', line=dict(dash='dot', color='lightgrey')
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=arima_conf.iloc[:,1], mode='lines',
        name='Upper CI', line=dict(dash='dot', color='lightgrey'),
        fill='tonexty', fillcolor='rgba(200,200,200,0.2)'
    ))
    fig.update_layout(title="ARIMA Forecast (Next 12 Months)", xaxis_title="Date", yaxis_title="Index")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
