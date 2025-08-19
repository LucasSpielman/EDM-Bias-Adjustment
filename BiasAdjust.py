# Install necessary libraries (uncomment and run these lines if needed)
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install xarray
# !pip install xclim[extras]
# !pip install openpyxl

# Import required libraries
import pandas as pd  # For handling tabular data
import xarray as xr  # For working with multi-dimensional arrays
import xclim.sdba as sdba  # For statistical bias adjustment
from xclim import set_options  # For configuring xclim options

# -------------------------------
# STEP 1: Load data from Excel
# -------------------------------
# Load observed and model data from Excel files.
# Observed data contains station data (e.g., Edmonton Blatchford).
# Model data contains climate model outputs.
# Both datasets must have a "date" column for time-based operations.

# Load observed data from Excel
obs_df = pd.read_excel("./Data/Edmonton_Blatchford_Joined.xlsx")

# Load model data from Excel
model_df = pd.read_excel("./Data/Edmonton_Tmin_Cannon_16_Compiled .xlsx")

# Convert 'date' columns to datetime format for easier time-based operations
obs_df["date"] = pd.to_datetime(obs_df["DATE"])  # Observed data
model_df["date"] = pd.to_datetime(model_df["ALL DATES"])  # Model data

# Replace anomalous values (-9999.9) with NaN
# These values are placeholders for missing or invalid data.
model_df = model_df.replace(-9999.9, pd.NA)
obs_df = obs_df.replace(-9999.9, pd.NA)

# Handle missing values
# Fill missing values with the column mean to ensure no gaps in the data.
model_df = model_df.fillna(model_df.mean())
obs_df = obs_df.fillna(obs_df.mean())

# Ensure all model columns are numeric
# Convert all columns to numeric, coercing invalid values to NaN.
model_df = model_df.apply(pd.to_numeric, errors="coerce")

# Handle missing values again (if needed)
model_df = model_df.fillna(model_df.mean())

# -------------------------------
# STEP 2: Convert all model columns to a single xarray.DataArray
# -------------------------------
# Combine all model columns (except the 'date' column) into a single DataArray.
# This allows us to perform bias adjustment on all columns simultaneously.

# Extract all model columns except 'date'
model_columns = [col for col in model_df.columns if col != "ALL DATES"]

# Create a DataArray for all model columns
model_tmin_all = xr.DataArray(
    model_df[model_columns].values.astype(float),  # Ensure numeric data
    dims=("time", "model"),  # Dimensions: time and model
    coords={
        "time": model_df["date"],  # Time coordinates
        "model": model_columns,  # Model names as coordinates
    },
    attrs={"units": "degC"},  # Metadata (units)
)

# Create a DataArray for observed Tmin (single column)
obs_tmin = xr.DataArray(
    obs_df["Tmin"].values,  # Data values
    dims="time",  # Dimension name
    coords={"time": obs_df["date"]},  # Time coordinates
    attrs={"units": "degC"},  # Metadata (units)
)

# -------------------------------
# STEP 3: Define training periods
# -------------------------------
# Training periods are time windows used to train the bias adjustment models.
# These periods are defined as slices of time.

train_slice_30yr = slice("1971-01-01", "2000-12-31")  # 30-year window
train_slice_50yr = slice("1951-01-01", "2000-12-31")  # 50-year window
train_slice_70yr = slice("1931-01-01", "2000-12-31")  # 70-year window

# =====================================================
# STEP 4: Train quantile mapping models
# =====================================================
# Quantile mapping is a statistical method to adjust biases in model data.
# It aligns the distribution of model data with observed data.
# xclim provides a built-in method called "EmpiricalQuantileMapping" (EQM).

# Check the time range of the data
# This ensures that the observed and model data overlap in time.
print("Observed time range:", obs_tmin["time"].min().values, "to", obs_tmin["time"].max().values)
print("Model time range:", model_tmin_all["time"].min().values, "to", model_tmin_all["time"].max().values)

# Ensure time coordinates are in datetime64 format
obs_tmin["time"] = pd.to_datetime(obs_tmin["time"].values)
model_tmin_all["time"] = pd.to_datetime(model_tmin_all["time"].values)

# Select data for the training period (30-year window)
obs_30 = obs_tmin.sel(time=train_slice_30yr)
mod_30 = model_tmin_all.sel(time=train_slice_30yr)

# Check if the selected data is empty
if obs_30.size == 0 or mod_30.size == 0:
    raise ValueError("The selected training period does not overlap with the data.")

# Align time steps
# Align observed and model data along the time dimension.
obs_30, mod_30 = xr.align(obs_30, mod_30, join="inner")

# Train the EQM model for the 30-year window
QM_30 = sdba.EmpiricalQuantileMapping.train(
    ref=obs_30,  # Observed data (reference)
    hist=mod_30,  # Model data (historical)
)

# Repeat the process for the 50-year window
obs_50 = obs_tmin.sel(time=train_slice_50yr)
mod_50 = model_tmin_all.sel(time=train_slice_50yr)
obs_50, mod_50 = xr.align(obs_50, mod_50, join="inner")
QM_50 = sdba.EmpiricalQuantileMapping.train(ref=obs_50, hist=mod_50)

# Repeat the process for the 70-year window
obs_70 = obs_tmin.sel(time=train_slice_70yr)
mod_70 = model_tmin_all.sel(time=train_slice_70yr)
obs_70, mod_70 = xr.align(obs_70, mod_70, join="inner")
QM_70 = sdba.EmpiricalQuantileMapping.train(ref=obs_70, hist=mod_70)

# =====================================================
# STEP 5: Apply corrections
# =====================================================
# Once trained, the EQM models can be applied to adjust all model data.
# This step corrects the model values for biases.

# Apply the 30-year EQM model to adjust Tmin
tmin_corrected_30 = QM_30.adjust(model_tmin_all)

# Apply the 50-year EQM model to adjust Tmin
tmin_corrected_50 = QM_50.adjust(model_tmin_all)

# Apply the 70-year EQM model to adjust Tmin
tmin_corrected_70 = QM_70.adjust(model_tmin_all)

# =====================================================
# STEP 6: Exceedance probability
# =====================================================
# Example: What is the probability that Tmin exceeds a threshold (e.g., 15°C)?
# We calculate this for raw model data and compare it with bias-adjusted results.

threshold = 15  # Threshold temperature in degrees Celsius

# Calculate the probability of exceeding the threshold for raw model data
prob_raw_all = (model_tmin_all > threshold).mean(dim="time")

# Calculate the probability for bias-adjusted data for each window
prob_corrected_30 = (tmin_corrected_30 > threshold).mean(dim="time")
prob_corrected_50 = (tmin_corrected_50 > threshold).mean(dim="time")
prob_corrected_70 = (tmin_corrected_70 > threshold).mean(dim="time")

# =====================================================
# STEP 7: Visualize results with Dash
# =====================================================
# We use Dash to create an interactive web app for visualizing the results.

import dash  # For creating web apps
from dash import dcc, html, dash_table  # For app components
import plotly.graph_objs as go  # For creating plots

# -------------------------------
# STEP 1: Slice data for plotting
# -------------------------------
# Define the time range for plotting
start_plot = "2025-01-01"
end_plot = "2030-12-31"

# Slice the raw and corrected data for the plotting range
model_subset = model_tmin_all.sel(time=slice(start_plot, end_plot))
tmin_30_subset = tmin_corrected_30.sel(time=slice(start_plot, end_plot))
tmin_50_subset = tmin_corrected_50.sel(time=slice(start_plot, end_plot))
tmin_70_subset = tmin_corrected_70.sel(time=slice(start_plot, end_plot))
obs_subset = obs_tmin.sel(time=slice(start_plot, end_plot))

# -------------------------------
# STEP 2: Create Dash app
# -------------------------------
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H2("Tmin Comparison: Raw vs Bias-Adjusted Models (2025-2030)"),  # Title
    dcc.Dropdown(  # Dropdown menu for selecting models
        id="model-dropdown",
        options=[{"label": model, "value": model} for model in model_columns],
        value=model_columns[0],  # Default value (first model)
        clearable=False,
    ),
    html.Div([
        html.Label("Exceedance Threshold (°C):"),
        dcc.Input(
            id="threshold-input",
            type="number",
            value=15,  # Default threshold
            step=0.1,
        ),
    ], style={"margin-bottom": "20px"}),
    dcc.Graph(id="tmin-graph"),  # Graph component
    html.H3("Exceedance Probabilities"),  # Table title
    dash_table.DataTable(  # Table to display probabilities
        id="probability-table",
        columns=[
            {"name": "Model", "id": "model"},
            {"name": "Raw", "id": "raw"},
            {"name": "Bias-Adjusted (30yr)", "id": "30yr"},
            {"name": "Bias-Adjusted (50yr)", "id": "50yr"},
            {"name": "Bias-Adjusted (70yr)", "id": "70yr"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
    )
])

# -------------------------------
# STEP 3: Define callbacks
# -------------------------------
@app.callback(
    [dash.dependencies.Output("tmin-graph", "figure"),
     dash.dependencies.Output("probability-table", "data")],
    [dash.dependencies.Input("model-dropdown", "value"),
     dash.dependencies.Input("threshold-input", "value")]
)
def update_graph_and_table(selected_model, threshold):
    # Select the data for the chosen model
    raw_data = model_subset.sel(model=selected_model)
    data_30yr = tmin_30_subset.sel(model=selected_model)
    data_50yr = tmin_50_subset.sel(model=selected_model)
    data_70yr = tmin_70_subset.sel(model=selected_model)

    # Get exceedance probabilities for the selected model
    prob_raw = (raw_data > threshold).mean().item()
    prob_30 = (data_30yr > threshold).mean().item()
    prob_50 = (data_50yr > threshold).mean().item()
    prob_70 = (data_70yr > threshold).mean().item()

    # Create the figure
    figure = {
        "data": [
            # Plot observed Tmin
            go.Scatter(
                x=obs_subset.time.values,
                y=obs_subset.values,
                mode="lines",
                name="Observed Tmin",
                line=dict(color="red", width=2, dash="dash")
            ),
            # Plot raw model Tmin
            go.Scatter(
                x=raw_data.time.values,
                y=raw_data.values,
                mode="lines",
                name=f"Raw Model Tmin ({selected_model})",
                line=dict(color="gray", width=2)
            ),
            # Plot 30-year bias-adjusted Tmin
            go.Scatter(
                x=data_30yr.time.values,
                y=data_30yr.values,
                mode="lines",
                name="Bias-Adjusted (30yr)",
                line=dict(color="blue", width=2)
            ),
            # Plot 50-year bias-adjusted Tmin
            go.Scatter(
                x=data_50yr.time.values,
                y=data_50yr.values,
                mode="lines",
                name="Bias-Adjusted (50yr)",
                line=dict(color="purple", width=2)
            ),
            # Plot 70-year bias-adjusted Tmin
            go.Scatter(
                x=data_70yr.time.values,
                y=data_70yr.values,
                mode="lines",
                name="Bias-Adjusted (70yr)",
                line=dict(color="cyan", width=2)
            )
        ],
        "layout": go.Layout(
            xaxis={"title": "Time"},  # X-axis label
            yaxis={"title": "Tmin (°C)"},  # Y-axis label
            template="plotly_white",  # Plot style
            hovermode="x unified"  # Unified hover mode
        )
    }

    # Create the table data
    table_data = [{
        "model": selected_model,
        "raw": f"{prob_raw:.3f}",
        "30yr": f"{prob_30:.3f}",
        "50yr": f"{prob_50:.3f}",
        "70yr": f"{prob_70:.3f}",
    }]

    return figure, table_data

# Run the app
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for easier troubleshooting