import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


df = pd.read_csv(r"C:\Users\User\Desktop\Data_visualization\queation_2\clean_fao_food_price_index.csv", parse_dates=['Date'])
df = df[['Date', 'Food Price Index']].dropna()
df = df.sort_values('Date')


prophet_df = df.rename(columns={'Date': 'ds', 'Food Price Index': 'y'})
m = Prophet(yearly_seasonality=True, daily_seasonality=False)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=24, freq='M')  # 2 years forecast
forecast = m.predict(future)

app = Dash(__name__)
app.title = "Food Price Forecast Dashboard"

# Layout (Z‑Pattern)
app.layout = html.Div([
    html.H1("📊 FAO Food Price Index Forecast Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label("Forecast Months Ahead"),
            dcc.Slider(
                id='month_slider',
                min=1, max=24, step=1,
                value=12,
                marks={i: str(i) for i in range(0,25,6)}
            ),
            html.Div(id='slider-output-container')
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        html.Div([
            dcc.Graph(id='forecast_graph')
        ], style={'width': '65%', 'display': 'inline-block'})
    ]),
    html.Div([
        html.H3("Key Insights"),
        html.P("✅ This dashboard uses a Z‑Pattern layout: controls on top-left, forecast chart top-right, and insights follow below."),
        html.P("✅ Use the slider to adjust how many months into the future to forecast.")
    ], style={'padding': '20px'})
])

# Callback for forecast update
@app.callback(
    Output('forecast_graph', 'figure'),
    Output('slider-output-container', 'children'),
    Input('month_slider', 'value')
)
def update_forecast(months):
    future_dynamic = m.make_future_dataframe(periods=months, freq='M')
    forecast_dynamic = m.predict(future_dynamic)

    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'], y=prophet_df['y'],
        mode='lines', name='Historical'
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dynamic['ds'], y=forecast_dynamic['yhat'],
        mode='lines', name='Forecast'
    ))
    # Confidence
    fig.add_trace(go.Scatter(
        x=forecast_dynamic['ds'], y=forecast_dynamic['yhat_upper'],
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dynamic['ds'], y=forecast_dynamic['yhat_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
        name='Confidence Interval'
    ))

    fig.update_layout(title="Forecast with Confidence Intervals",
                      xaxis_title="Date", yaxis_title="Food Price Index",
                      template='plotly_white')

    return fig, f"Showing forecast for {months} months ahead."

if __name__ == '__main__':
    app.run(debug=True)
