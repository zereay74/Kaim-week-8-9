import dash
from dash import dcc, html  # ✅ Updated import
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import requests

# Flask API URL
API_URL = "http://127.0.0.1:5000"

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for deployment

# Fetch fraud summary data
fraud_summary_response = requests.get(f"{API_URL}/summary").json()

# Fetch fraud by country data
fraud_by_country_response = requests.get(f"{API_URL}/fraud_by_country").json()
fraud_by_country_df = pd.DataFrame(fraud_by_country_response)

# Fetch fraud by device & browser data
try:
    fraud_by_device_browser_response = requests.get(f"{API_URL}/fraud_by_device_browser").json()
    fraud_by_device_browser_df = pd.DataFrame(fraud_by_device_browser_response)
except requests.exceptions.RequestException as e:
    fraud_by_device_browser_df = pd.DataFrame(columns=["device_id", "browser", "fraud_count"])
    print(f"API Error (fraud_by_device_browser): {e}")

# Fetch fraud trends over time data
try:
    fraud_trends_response = requests.get(f"{API_URL}/fraud_trends").json()
    fraud_trends_df = pd.DataFrame(fraud_trends_response)

    # Ensure correct datetime format
    fraud_trends_df["date"] = pd.to_datetime(fraud_trends_df["date"])
except requests.exceptions.RequestException as e:
    fraud_trends_df = pd.DataFrame(columns=["date", "fraud_cases"])
    print(f"API Error (fraud_trends): {e}")

# Layout
app.layout = dbc.Container([
    html.H1("Fraud Detection Dashboard", className="text-center mb-4"),

    # Fraud Cases by Device & Browser
    dbc.Row([
        dbc.Col([
            html.H3("Fraud Cases by Device & Browser"),
            dcc.Graph(
                figure=px.bar(
                    fraud_by_device_browser_df,
                    x="device_id",
                    y="fraud_count",
                    color="browser",
                    title="Fraud Cases per Device & Browser",
                    barmode="group"
                ) if not fraud_by_device_browser_df.empty else {},
                id="fraud-device-browser-bar"
            )
        ])
    ], className="mb-4"),

    # Summary Statistics
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4("Total Transactions", className="card-title text-center"),
            html.H2(f"{fraud_summary_response['total_transactions']:,}", className="text-center")
        ], body=True), width=4),

        dbc.Col(dbc.Card([
            html.H4("Total Fraud Cases", className="card-title text-center"),
            html.H2(f"{fraud_summary_response['fraud_cases']:,}", className="text-center")
        ], body=True), width=4),

        dbc.Col(dbc.Card([
            html.H4("Fraud Percentage", className="card-title text-center"),
            html.H2(f"{fraud_summary_response['fraud_percentage']:.2f}%", className="text-center")
        ], body=True), width=4),
    ], className="mb-4"),

    # Fraud Cases Over Time (Fixed Line Chart)
    dbc.Row([
        dbc.Col([
            html.H3("Fraud Cases Over Time"),
            dcc.Graph(
                figure=px.line(
                    fraud_trends_df,
                    x="date",
                    y="fraud_cases",  # ✅ Fixed column name
                    title="Fraud Cases Over Time",
                    markers=True
                ) if not fraud_trends_df.empty else {},
                id="fraud-trends-line"
            )
        ])
    ], className="mb-4"),

    # Fraud Cases by Country (Choropleth Map)
    dbc.Row([
        dbc.Col([
            html.H3("Fraud Cases by Country"),
            dcc.Graph(
                figure=px.choropleth(
                    fraud_by_country_df,
                    locations="country",
                    locationmode="country names",
                    color="fraud_count",
                    title="Fraud Cases by Country",
                    color_continuous_scale="Reds"
                ) if not fraud_by_country_df.empty else {},
                id="fraud-country-map"
            )
        ])
    ], className="mb-4")

])

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
