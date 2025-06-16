import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from prophet import Prophet
from prophet.plot import plot_components_plotly

# 🔧 Page config: Compact layout
st.set_page_config(page_title="🔋 PJM Forecast", layout="centered")

# Title
st.title("🔮 PJM Energy Demand Forecasting")

# Slider
n_days = st.slider("📅 Select forecast horizon (days)", 1, 30, 7)

# Load model
with open("prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and prepare historical data
df_hist = pd.read_csv("cleaned_pjm.csv", parse_dates=["Datetime"])
df_hist_daily = df_hist.resample("D", on="Datetime").mean().dropna()
last_date = df_hist_daily.index[-1]

# Create future dates and forecast
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
future = pd.DataFrame({'ds': future_dates})
forecast = model.predict(future)

# Prepare forecast table
forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_result.set_index('ds', inplace=True)
forecast_result.rename(columns={
    'yhat': 'Forecast',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
}, inplace=True)

# 📋 Show forecast table first
st.subheader("📋 Forecast Table")
st.dataframe(forecast_result.style.format("{:.2f}"))

# 📈 Fancy forecast plot
st.subheader("📊 Forecast Trend (with Confidence Interval)")
fig = go.Figure()

# Forecast line
fig.add_trace(go.Scatter(
    x=forecast_result.index,
    y=forecast_result['Forecast'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='royalblue', width=3),
    marker=dict(size=6)
))

# Confidence interval shaded area
fig.add_trace(go.Scatter(
    x=forecast_result.index.tolist() + forecast_result.index[::-1].tolist(),
    y=forecast_result['Upper Bound'].tolist() + forecast_result['Lower Bound'][::-1].tolist(),
    fill='toself',
    fillcolor='rgba(65,105,225,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name='Confidence Interval',
    showlegend=True
))

# Add recent actuals
recent_actual = df_hist_daily[-7:]
fig.add_trace(go.Scatter(
    x=recent_actual.index,
    y=recent_actual.iloc[:, 0],
    mode='lines+markers',
    name='Recent Actual',
    line=dict(color='green', dash='dot'),
    marker=dict(size=6)
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="MW",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", x=0.1, y=1.1),
    margin=dict(t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# 🧠 Interpretation
st.markdown("### 🧠 Forecast Insights")
start_forecast = forecast_result.iloc[0]
end_forecast = forecast_result.iloc[-1]

st.success(f"""
- 📅 Forecast starts on **{start_forecast.name.date()}** with an expected demand of **{start_forecast['Forecast']:.2f} MW**.
- 📅 Forecast ends on **{end_forecast.name.date()}** with a predicted demand of **{end_forecast['Forecast']:.2f} MW**.
- 🟦 The blue shaded region shows the **95% confidence interval**, reflecting uncertainty.
- 📈 Useful for **energy planning, demand estimation, and resource allocation**.
""")

# 📆 Seasonal Components
st.subheader("🧭 Trend & Seasonality")
st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

# 📥 Download forecast CSV
csv = forecast_result.to_csv().encode('utf-8')
st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
