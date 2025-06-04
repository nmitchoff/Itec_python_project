# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred

# ------------------------------
#  Page configuration must be first
# ------------------------------
st.set_page_config(page_title="CPI & Store Count Explorer", layout="wide")

# ------------------------------
#  Helper functions using new caching APIs
# ------------------------------

@st.cache_data(show_spinner=False)
def load_starbucks_data(csv_path: str) -> pd.DataFrame:
    """
    Load Starbucks data from CSV, parse dates, and set index to quarter-end timestamps.
    Returns a DataFrame with a DateTimeIndex at quarter-ends.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    # Convert each date to Period('Q') then back to timestamp at quarter-end
    df = df.set_index(pd.DatetimeIndex(df["date"]).to_period("Q").to_timestamp("Q"))
    df = df.sort_index()
    return df

@st.cache_data(show_spinner=False)
def fetch_quarterly_cpi(fred_api_key: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch CPIAUCSL (Consumer Price Index for All Urban Consumers) from FRED between
    start_date and end_date (YYYY-MM-DD). Resample monthly data to quarterly averages,
    and return a pd.Series indexed by quarter-end timestamps.
    """
    fred = Fred(api_key=fred_api_key)
    cpi_monthly = fred.get_series("CPIAUCSL", observation_start=start_date, observation_end=end_date)
    cpi_monthly.index = pd.to_datetime(cpi_monthly.index)
    # Resample to quarterly frequency by taking the average of each quarter
    cpi_q = cpi_monthly.resample("Q").mean()
    cpi_q.index = pd.DatetimeIndex(cpi_q.index.to_period("Q").to_timestamp("Q"))
    return cpi_q.rename("CPI")

# ------------------------------
#  Streamlit App Layout
# ------------------------------
st.title("CPI & Starbucks Store Count Explorer")

st.markdown(
    """
    Use the sliders below to project future CPI (Consumer Price Index) and Starbucks store count.
    Historical CPI is fetched from FRED using your API key stored in Streamlit secrets (`fred_api_key`).
    Historical store count is loaded from the Starbucks quarterly dataset.
    Adjust the annual growth rates and forecast horizon to visualize how CPI and store count 
    might evolve in the future.
    """
)

# ------------------------------
#  Sidebar: User Inputs
# ------------------------------
st.sidebar.header("Forecast Settings")

# 1. Forecast horizon in years
forecast_years = st.sidebar.slider(
    label="Forecast Horizon (years)",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
    help="Select how many years into the future to project CPI and store count."
)

# 2. CPI annual growth rate (%)
cpi_growth_pct = st.sidebar.slider(
    label="CPI Annual Growth Rate (%)",
    min_value=-5.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Assumed annual growth rate for CPI (e.g., inflation)."
)

# 3. Store count annual growth rate (%)
store_growth_pct = st.sidebar.slider(
    label="Store Count Annual Growth Rate (%)",
    min_value=-5.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
    help="Assumed annual growth rate for Starbucks store count."
)

# ------------------------------
#  Data Loading and Preparation
# ------------------------------
# Load Starbucks data (must have 'date' and 'store_count' columns)
csv_path = "final_project_starbucks_data.csv"
with st.spinner("Loading Starbucks data..."):
    try:
        sbux_df = load_starbucks_data(csv_path)
    except FileNotFoundError:
        st.error(f"Could not find CSV at path: '{csv_path}'. Please verify the filename/location.")
        st.stop()

# Extract historical store count series
store_series = sbux_df["store_count"].copy()

# Determine data date range for CPI fetch
start_date = sbux_df.index.min().strftime("%Y-%m-%d")
end_date = sbux_df.index.max().strftime("%Y-%m-%d")

# Fetch CPI series from FRED, using API key from Streamlit secrets
fred_api_key = st.secrets.get("fred_api_key", "")
if not fred_api_key:
    st.error("❌ No FRED API key found in Streamlit secrets. Please add `fred_api_key` to .streamlit/secrets.toml.")
    st.stop()

with st.spinner("Fetching CPI data from FRED..."):
    try:
        cpi_series = fetch_quarterly_cpi(fred_api_key, start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching CPI data: {e}")
        st.stop()

# ------------------------------
#  Build Forecast Index
# ------------------------------
# Convert years to number of quarters
forecast_quarters = forecast_years * 4

# Last observed quarter-end date
last_date = store_series.index.max()

# Future quarter-end index (datetime)
future_index = pd.date_range(
    start=(last_date + pd.tseries.offsets.QuarterEnd(1)),
    periods=forecast_quarters,
    freq="Q"
)

# ------------------------------
#  Project CPI
# ------------------------------
# Last observed CPI value
last_cpi = cpi_series.iloc[-1]

# Convert annual growth rate (%) to quarterly multiplier
quarterly_cpi_multiplier = (1 + cpi_growth_pct / 100) ** (1 / 4)

# Generate CPI projection
proj_cpi_values = [
    last_cpi * (quarterly_cpi_multiplier ** i) for i in range(1, forecast_quarters + 1)
]
proj_cpi = pd.Series(data=proj_cpi_values, index=future_index, name="Projected CPI")

# ------------------------------
#  Project Store Count
# ------------------------------
# Last observed store count
last_store = store_series.iloc[-1]

# Convert annual growth rate (%) to quarterly multiplier
quarterly_store_multiplier = (1 + store_growth_pct / 100) ** (1 / 4)

# Generate store count projection
proj_store_values = [
    last_store * (quarterly_store_multiplier ** i) for i in range(1, forecast_quarters + 1)
]
proj_store = pd.Series(data=proj_store_values, index=future_index, name="Projected Store Count")

# ------------------------------
#  Plotting: CPI
# ------------------------------
st.subheader("CPI: Historical vs. Projected")

fig_cpi, ax_cpi = plt.subplots(figsize=(10, 4))
# Plot historical CPI
ax_cpi.plot(
    cpi_series.index,
    cpi_series.values,
    marker="o",
    linestyle="-",
    color="tab:blue",
    label="Historical CPI"
)
# Plot projected CPI
ax_cpi.plot(
    proj_cpi.index,
    proj_cpi.values,
    marker="o",
    linestyle="--",
    color="tab:orange",
    label=f"Projected CPI (Annual Growth: {cpi_growth_pct:.1f}%)"
)

ax_cpi.set_title("Consumer Price Index (CPI) by Quarter")
ax_cpi.set_xlabel("Quarter")
ax_cpi.set_ylabel("CPI (Index Value)")
ax_cpi.legend()
ax_cpi.grid(True)

st.pyplot(fig_cpi)

# ------------------------------
#  Plotting: Store Count
# ------------------------------
st.subheader("Starbucks Store Count: Historical vs. Projected")

fig_store, ax_store = plt.subplots(figsize=(10, 4))
# Plot historical store count
ax_store.plot(
    store_series.index,
    store_series.values,
    marker="o",
    linestyle="-",
    color="tab:green",
    label="Historical Store Count"
)
# Plot projected store count
ax_store.plot(
    proj_store.index,
    proj_store.values,
    marker="o",
    linestyle="--",
    color="tab:red",
    label=f"Projected Store Count (Annual Growth: {store_growth_pct:.1f}%)"
)

ax_store.set_title("Starbucks Store Count by Quarter")
ax_store.set_xlabel("Quarter")
ax_store.set_ylabel("Store Count")
ax_store.legend()
ax_store.grid(True)

st.pyplot(fig_store)

# ------------------------------
#  Display Data Tables (Optional)
# ------------------------------
with st.expander("Show underlying data tables"):
    st.write("Historical CPI (Quarterly):")
    st.dataframe(cpi_series.to_frame())

    st.write("Projected CPI:")
    st.dataframe(proj_cpi.to_frame())

    st.write("Historical Store Count:")
    st.dataframe(store_series.to_frame())

    st.write("Projected Store Count:")
    st.dataframe(proj_store.to_frame())
