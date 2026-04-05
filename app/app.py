from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

"""
Load Data and Models:
- SARIMAX model is loaded using joblib from the `../data/models/sarimax_model_hourly.pkl` file.
- Test data (`test_data.csv`) and exogenous data (`exog_test_hourly.csv`) are loaded as pandas DataFrames.
- Naive model data (`naive_data.csv`) is loaded to act as a baseline for comparison.
- Index alignment ensures all data sources have a common datetime index.
"""

# Load the SARIMAX model
fitted_model = joblib.load("../data/models/sarimax_model_hourly.pkl")

# Load test data
test_data = pd.read_csv(
    "../data/test_data/test_data.csv", parse_dates=["Date"], index_col="Date"
)
y_test = test_data["Carbon Intensity gCO₂eq/kWh (LCA)"]

# Load exogenous test data
exog_test = pd.read_csv(
    "../data/test_data/exog_test_hourly.csv", parse_dates=["Date"], index_col="Date"
)
exog_test = exog_test.dropna()

# load naive model
naive_data = pd.read_csv(
    "../data/models/naive_data.csv", parse_dates=["Date"], index_col="Date"
)
naive_model = naive_data["Carbon Intensity gCO₂eq/kWh (LCA)"]

# Ensure indices are aligned
if not exog_test.index.equals(y_test.index):
    common_index = exog_test.index.intersection(y_test.index)
    exog_test = exog_test.loc[common_index]
    y_test = y_test.loc[common_index]


# model predicts
forecast = fitted_model.forecast(steps=len(y_test), exog=exog_test)
forecast_series = pd.Series(forecast)

# Error Analysis
residuals = y_test - forecast_series
residuals_naive = y_test - naive_model

# Compute global metrics
mae = mean_absolute_error(y_test, forecast_series)
rmse = np.sqrt(mean_squared_error(y_test, forecast_series))
r2 = r2_score(y_test, forecast_series)

# Create Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div(
    [
        # Student's Name and School Info
        html.Div(
            children=[
                html.P(
                    html.B("Millena Kidane"),
                    style={
                        "textAlign": "left",
                        "fontSize": "16px",
                        "marginBottom": "1px",
                    },
                ),
                html.P(
                    "Haute École ARC Ingénierie",
                    style={
                        "textAlign": "left",
                        "fontSize": "16px",
                        "marginBottom": "20px",
                    },
                ),
            ],
        ),
        # School Logo
        html.Div(
            children=[
                html.Img(
                    src="/assets/logo.PNG",
                    style={
                        "height": "100px",
                        "display": "block",
                        "marginLeft": "left",
                        "marginRight": "left",
                    },
                )
            ],
            style={"marginBottom": "20px"},
        ),
        # Footer with "Tous droits réservés"
        html.Div(
            children=[
                html.P(
                    "Tous droits réservés",
                    style={"textAlign": "left", "fontSize": "16px", "color": "gray"},
                )
            ],
            style={"marginTop": "20px"},
        ),
        html.H1(
            "Forecast Dashboard",
            style={
                "textAlign": "center",
                "fontSize": "40px",
                "marginBottom": "20px",
                "color": "brown",
            },
        ),
        # Description Section
        html.Div(
            children=[
                html.P(
                    "This dashboard provides the carbon intensity of Switzerland for the year 2023 by a trained model."
                    "The graphs show comparative of a forecasted and actual carbon intensity levels."
                    "The data is trained on the carbon Intensity levels, production and consumption of energy"
                    " for the years 2022 and 2023 for the year 2021 and 2022.",
                    style={
                        "textAlign": "left",
                        "fontSize": "21px",
                        "color": "black",
                    },
                ),
                html.P(
                    "Here the model used is a SARIMAX model. The dashboard shows "
                    "different aggregation levels like hourly, daily and weekly carbon Intensity data of predicted data."
                    "We can see model performance via different metrics used here below.",
                    style={
                        "textAlign": "left",
                        "fontSize": "21px",
                        "color": "black",
                    },
                ),
            ],
            style={"margin": "20px 0"},
        ),
        # Dropdown for aggregation level
        html.Label("Select Aggregation Level:"),
        dcc.Dropdown(
            id="aggregation-level",
            options=[
                {"label": "Daily", "value": "D"},
                {"label": "Weekly", "value": "W"},
                {"label": "Monthly", "value": "ME"},
            ],
            value="ME",
            style={"width": "50%", "fontSize": "20px"},
        ),
        # Forcast vs actual
        html.Div(
            children=[
                html.H3("Forecast vs Actual"),
                html.P(
                    "Here the graph shows the model performing much better than the naive model "
                    "On all aggregation levels.",
                    style={"textAlign": "left", "fontSize": "18px", "color": "black"},
                ),
                # Graph for forecast vs actual
                dcc.Graph(id="forecast-graph"),
            ],
            style={"margin": "20px 0", "color": "brown"},
        ),
        # Add the residuals graph here
        html.Div(
            children=[
                html.H3("Error Analysis"),
                html.P(
                    "Here we can see that the naive model prediction is much larger "
                    "than the actual values. Whereas the model is clse to zero (similar to the actual data)",
                    style={"textAlign": "left", "fontSize": "18px", "color": "black"},
                ),
                dcc.Graph(id="residuals-graph"),
            ],
            style={"margin": "20px 0", "color": "brown"},
        ),
        # Add the residual histogram
        html.Div(
            children=[
                html.H3("Error histogram"),
                html.P(
                    "This shows the error values on x-axis and the frequency of errors on y axis"
                    "Ths naive model predicts the values higher than the actual data So large -ve error values."
                    "But the trained model errors are closer to 0",
                    style={"textAlign": "left", "fontSize": "18px", "color": "black"},
                ),
                dcc.Graph(
                    id="residuals-histogram"
                ),  # Histogram for residual distribution
            ],
            style={"margin": "20px 0", "color": "brown"},
        ),
        # Error summary
        html.Div(
            children=[
                html.H3("Error summary"),
                html.P(
                    "Summary statistic of the model",
                    style={"textAlign": "left", "fontSize": "18px", "color": "black"},
                ),
                html.Div(id="error-summary", style={"margin": "20px"}),
            ]
        ),
    ],
    style={"color": "brown", "padding": "20px"},
)


# Callback to update the graph dynamically
@app.callback(Output("forecast-graph", "figure"), Input("aggregation-level", "value"))
def update_forecast(aggregation_level):
    """
    Callback: update_forecast(aggregation_level)
    - Updates the "Forecast vs Actual" graph dynamically based on the selected aggregation level.
    - Aggregates the forecasted, actual, and naive model data using the specified level.
    - Recomputes global metrics (MAE, RMSE, R²) for the aggregated data.
    - Returns a Plotly figure with forecast, actual, and naive model lines.
    """

    # Aggregate forecast and actual data
    aggregated_forecast = forecast_series.resample(aggregation_level).mean()
    aggregated_actual = y_test.resample(aggregation_level).mean()
    aggregated_naive = naive_model.resample(aggregation_level).mean()

    # Recompute metrics for aggregated data
    mae = mean_absolute_error(aggregated_actual, aggregated_forecast)
    rmse = np.sqrt(mean_squared_error(aggregated_actual, aggregated_forecast))
    r2 = r2_score(aggregated_actual, aggregated_forecast)

    # Create the graph
    figure = {
        "data": [
            # Actual data line
            go.Scatter(
                x=aggregated_actual.index,
                y=aggregated_actual,
                mode="lines+markers",
                name=f"Actual ({aggregation_level})",
            ),
            # Forecast or predicted  data line
            go.Scatter(
                x=aggregated_forecast.index,
                y=aggregated_forecast,
                mode="lines+markers",
                name=f"Forecast ({aggregation_level})",
                line=dict(dash="dash"),
            ),
            # Naive data line
            go.Scatter(
                x=aggregated_naive.index,
                y=aggregated_naive,
                mode="lines+markers",
                name=f"Forecast(naive) ({aggregation_level})",
                line=dict(dash="dot", color="green"),
            ),
        ],
        "layout": {
            "title": f"Forecast vs Actual Carbon Intensity 2023 ({aggregation_level})",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Carbon Intensity (gCO₂eq/kWh)"},
            "template": "plotly_white",
            "annotations": [
                {
                    "text": f"<b>MAE:</b> {mae:.2f}<br><b>RMSE:</b> {rmse:.2f}<br><b>R²:</b> {r2:.2f}",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 1.1,
                    "y": 0.6,
                    "showarrow": False,
                    "align": "center",
                    "font": {"size": 18, "color": "brown"},
                }
            ],
        },
    }
    return figure


# Callback to update the residuals graph
@app.callback(Output("residuals-graph", "figure"), Input("aggregation-level", "value"))
def update_residuals(aggregation_level):
    """
    Callback: update_residuals(aggregation_level)
    - Updates the residuals graph dynamically based on the selected aggregation level.
    - Aggregates residuals for the SARIMAX model and the naive model.
    - Returns a Plotly figure with residual lines for both models.
    """

    # Aggregate residuals based on the selected aggregation level
    aggregated_residuals = residuals.resample(aggregation_level).mean()
    aggregated_residuals_naive = residuals_naive.resample(aggregation_level).mean()

    # Create the residuals figure
    figure = {
        "data": [
            go.Scatter(
                x=aggregated_residuals.index,
                y=aggregated_residuals,
                mode="lines+markers",
                name="error model",
            ),
            go.Scatter(
                x=aggregated_residuals_naive.index,
                y=aggregated_residuals_naive,
                mode="lines+markers",
                name="error naive",
            ),
        ],
        "layout": {
            "title": "Error Analysis",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Residuals"},
            "template": "plotly_white",
        },
    }
    return figure


# Optional: Callback to update the residual histogram
@app.callback(
    Output("residuals-histogram", "figure"), Input("aggregation-level", "value")
)
def update_residual_histogram(aggregation_level):
    """
    Callback: update_residual_histogram(aggregation_level)
    - Updates the histogram of errors dynamically based on the selected aggregation level.
    - Aggregates errors for the SARIMAX model and the naive model.
    - Returns a Plotly histogram comparing the error distributions of both models.
    """

    aggregated_residuals = residuals.resample(aggregation_level).mean()
    aggregated_residuals_naive = residuals_naive.resample(aggregation_level).mean()
    # Create a histogram of residuals
    figure = {
        "data": [
            go.Histogram(x=aggregated_residuals, nbinsx=130, name="error distribution"),
            go.Histogram(
                x=aggregated_residuals_naive,
                nbinsx=130,
                name="error distribution naive",
            ),
        ],
        "layout": {
            "title": "Distribution of errors by trained model vs naive model",
            "xaxis": {"title": "error value"},
            "yaxis": {"title": "Frequency of errors"},
            "template": "plotly_white",
        },
    }
    return figure


# Callback to update the error summary
@app.callback(Output("error-summary", "children"), Input("aggregation-level", "value"))
def update_error_summary(aggregation_level):
    """
    Callback: update_error_summary(aggregation_level)
    - Computes summary statistics (mean, standard deviation, max, min) for residuals based on the selected aggregation level.
    - Returns a formatted HTML block with the computed statistics.
    """

    # Calculate summary statistics
    aggregated_residuals = residuals.resample(aggregation_level).mean()
    mean_residual = aggregated_residuals.mean()
    std_residual = aggregated_residuals.std()
    max_residual = aggregated_residuals.max()
    min_residual = aggregated_residuals.min()

    # Return as a formatted HTML block
    return html.Div(
        [
            html.H4("Error Summary:"),
            html.P(f"Mean Residual: {mean_residual:.2f}"),
            html.P(f"Standard Deviation: {std_residual:.2f}"),
            html.P(f"Max Residual: {max_residual:.2f}"),
            html.P(f"Min Residual: {min_residual:.2f}"),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True)
