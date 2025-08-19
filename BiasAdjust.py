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
# We assume the data is stored in two Excel files:
#   - "obs" (observed data) contains station data (e.g., Edmonton Blatchford)
#   - "model" (model data) contains climate model outputs
#
# Each file should have columns like ["date", "Tmax", "Tmin"].
# Dates must be converted to datetime format for compatibility with xclim.

# Load observed data from Excel
obs_df = pd.read_excel("./Data/Edmonton_Blatchford_Joined.xlsx")

# Load model data from Excel
model_df = pd.read_excel("./Data/Edmonton_Tmin_Cannon_16_Compiled .xlsx")

# Convert 'date' columns to datetime format for easier time-based operations
obs_df["date"] = pd.to_datetime(obs_df["DATE"])  # Observed data
model_df["date"] = pd.to_datetime(model_df["ALL DATES"])  # Model data

# -------------------------------
# STEP 2: Convert to xarray DataArrays
# -------------------------------
# xclim works best with xarray DataArrays, which are time-indexed arrays.
# Here, we create separate DataArrays for Tmin (minimum temperature).
# Note: You can repeat this process for Tmax (maximum temperature) if needed.

# Create a DataArray for observed Tmin
obs_tmin = xr.DataArray(
    obs_df["Tmin"].values,  # Data values
    dims="time",  # Dimension name
    coords={"time": obs_df["date"]},  # Time coordinates
    attrs={"units": "degC"},  # Metadata (units)
)

# Create a DataArray for model Tmin
model_tmin = xr.DataArray(
    model_df["tasmin_FGOALS-g3_historical_ssp245_(degC)"].values,
    dims="time",
    coords={"time": model_df["date"]},
    attrs={"units": "degC"},
)

# -------------------------------
# STEP 3: Define training periods
# -------------------------------
# Training periods are time windows used to train the bias adjustment models.
# Here, we define 30-year, 50-year, and 70-year windows.

train_slice_30yr = slice("1971-01-01", "2000-12-31")  # 30-year window
train_slice_50yr = slice("1951-01-01", "2000-12-31")  # 50-year window
train_slice_70yr = slice("1931-01-01", "2000-12-31")  # 70-year window

# =====================================================
# STEP 4: Train quantile mapping models
# =====================================================
# Quantile mapping is a statistical method to adjust biases in model data.
# It aligns the distribution of model data with observed data.
# xclim provides a built-in method called "EmpiricalQuantileMapping" (EQM).

# Align observed and model data for the 50-year window
obs_30 = obs_tmin.sel(time=train_slice_50yr)  # Select observed data
mod_30 = model_tmin.sel(time=train_slice_50yr)  # Select model data
obs_30, mod_30 = xr.align(obs_30, mod_30, join="inner")  # Align time steps

# Train the EQM model for the 30-year window
QM_30 = sdba.EmpiricalQuantileMapping.train(
    ref=obs_30,  # Observed data (reference)
    hist=mod_30,  # Model data (historical)
)

# Repeat the process for the 50-year window
obs_50 = obs_tmin.sel(time=train_slice_50yr)
mod_50 = model_tmin.sel(time=train_slice_50yr)
obs_50, mod_50 = xr.align(obs_50, mod_50, join="inner")
QM_50 = sdba.EmpiricalQuantileMapping.train(ref=obs_50, hist=mod_50)

# Repeat the process for the 70-year window
obs_70 = obs_tmin.sel(time=train_slice_70yr)
mod_70 = model_tmin.sel(time=train_slice_70yr)
obs_70, mod_70 = xr.align(obs_70, mod_70, join="inner")
QM_70 = sdba.EmpiricalQuantileMapping.train(ref=obs_70, hist=mod_70)

# =====================================================
# STEP 5: Apply corrections
# =====================================================
# Once trained, the EQM models can be applied to adjust all model data.
# This step corrects the model values for biases.

# Apply the 30-year EQM model to adjust Tmin
tmin_corrected_30 = QM_30.adjust(model_tmin)

# Apply the 50-year EQM model to adjust Tmin
tmin_corrected_50 = QM_50.adjust(model_tmin)

# Apply the 70-year EQM model to adjust Tmin
tmin_corrected_70 = QM_70.adjust(model_tmin)

# =====================================================
# STEP 6: Exceedance probability
# =====================================================
# Example: What is the probability that Tmin exceeds a threshold (e.g., 15°C)?
# We calculate this for raw model data and compare it with bias-adjusted results.

threshold = 15  # Threshold temperature in degrees Celsius

# Calculate the probability of exceeding the threshold for raw model data
prob_raw = (model_tmin > threshold).mean().item()

# Calculate the probability for bias-adjusted data (30-year window)
prob_corrected_30 = (tmin_corrected_30 > threshold).mean().item()

# Calculate the probability for bias-adjusted data (50-year window)
prob_corrected_50 = (tmin_corrected_50 > threshold).mean().item()

# Calculate the probability for bias-adjusted data (70-year window)
prob_corrected_70 = (tmin_corrected_70 > threshold).mean().item()

# Print the results
print(f"Raw exceedance probability: {prob_raw:.3f}")
print(f"Bias-adjusted (30yr window): {prob_corrected_30:.3f}")
print(f"Bias-adjusted (50yr window): {prob_corrected_50:.3f}")
print(f"Bias-adjusted (70yr window): {prob_corrected_70:.3f}")

# =====================================================
# STEP 7: Visualize results with Dash
# =====================================================
# We use Dash to create an interactive web app for visualizing the results.

import dash  # For creating web apps
from dash import dcc, html  # For app components
import plotly.graph_objs as go  # For creating plots

# -------------------------------
# STEP 1: Slice data for plotting
# -------------------------------
# Define the time range for plotting
start_plot = "2025-01-01"
end_plot = "2030-12-31"

# Slice the raw and corrected data for the plotting range
model_subset = model_tmin.sel(time=slice(start_plot, end_plot))
tmin_30_subset = tmin_corrected_30.sel(time=slice(start_plot, end_plot))
tmin_50_subset = tmin_corrected_50.sel(time=slice(start_plot, end_plot))
tmin_70_subset = tmin_corrected_70.sel(time=slice(start_plot, end_plot))

# -------------------------------
# STEP 2: Create Dash app
# -------------------------------
# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H2("Tmin Comparison: Raw vs Bias-Adjusted (2025-2030)"),  # Title
    dcc.Graph(  # Graph component
        id="tmin-graph",
        figure={
            "data": [
                # Plot raw model Tmin
                go.Scatter(
                    x=model_subset.time.values,
                    y=model_subset.values,
                    mode="lines",
                    name="Raw Model Tmin",
                    line=dict(color="gray", width=2)
                ),
                # Plot bias-adjusted Tmin (30-year window)
                go.Scatter(
                    x=tmin_30_subset.time.values,
                    y=tmin_30_subset.values,
                    mode="lines",
                    name="Bias-Adjusted (30yr)",
                    line=dict(color="blue", width=2)
                ),
                # Plot bias-adjusted Tmin (50-year window)
                go.Scatter(
                    x=tmin_50_subset.time.values,
                    y=tmin_50_subset.values,
                    mode="lines",
                    name="Bias-Adjusted (50yr)",
                    line=dict(color="green", width=2)
                ),
                # Plot bias-adjusted Tmin (70-year window)
                go.Scatter(
                    x=tmin_70_subset.time.values,
                    y=tmin_70_subset.values,
                    mode="lines",
                    name="Bias-Adjusted (70yr)",
                    line=dict(color="red", width=2)
                )
            ],
            "layout": go.Layout(
                xaxis={"title": "Time"},  # X-axis label
                yaxis={"title": "Tmin (°C)"},  # Y-axis label
                template="plotly_white",  # Plot style
                hovermode="x unified"  # Unified hover mode
            )
        }
    )
])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for easier troubleshooting