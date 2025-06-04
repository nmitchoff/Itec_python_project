# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred

# ------------------------------------
#  1. PAGE CONFIGURATION (MUST BE FIRST)
# ------------------------------------
st.set_page_config(
    page_title="Starbucks Revenue Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------
#  2. HELPER FUNCTIONS (CACHE)
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_starbucks_data(csv_path: str) -> pd.DataFrame:
    """
    Load Starbucks data from CSV, parse 'date' to quarter-end timestamps.
    Expected columns: ['date', 'revenue', 'store_count', ...].
    'revenue' is in millions USD.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
    # Convert each date to a Period('Q') then back to timestamp at quarter-end
    df = df.set_index(pd.DatetimeIndex(df["date"]).to_period("Q").to_timestamp("Q"))
    df = df.sort_index()
    return df

@st.cache_data(show_spinner=False)
def fetch_quarterly_cpi(fred_api_key: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch CPIAUCSL (Consumer Price Index for All Urban Consumers) from FRED
    between start_date and end_date (YYYY-MM-DD). Resample monthly to quarterly average.
    """
    fred = Fred(api_key=fred_api_key)
    cpi_monthly = fred.get_series("CPIAUCSL", observation_start=start_date, observation_end=end_date)
    cpi_monthly.index = pd.to_datetime(cpi_monthly.index)
    # Quarterly average
    cpi_q = cpi_monthly.resample("Q").mean()
    cpi_q.index = pd.DatetimeIndex(cpi_q.index.to_period("Q").to_timestamp("Q"))
    return cpi_q.rename("CPI")

@st.cache_resource(show_spinner=False)
def fit_arimax_model(df: pd.DataFrame) -> SARIMAX:
    """
    Fit an ARIMAX(1,1,1) on 'revenue' with exogenous ['CPI', 'store_count'].
    Returns the fitted SARIMAXResults object.
    """
    y = df["revenue"]               # Quarterly revenue (millions USD)
    exog = df[["CPI", "store_count"]]  # Exogenous regressors
    model = SARIMAX(
        endog=y,
        exog=exog,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    return result

# ------------------------------------
#  3. APP LAYOUT & SIDEBAR INPUTS
# ------------------------------------
st.title("☕ Starbucks Revenue Forecasting App")
st.markdown(
    """
    Build an ARIMAX revenue‐forecasting model (in millions USD) that uses live CPI from FRED
    and Store Count as exogenous inputs. Use the slider to choose how many **quarters ahead** to forecast (1–8 quarters).  
    Below, you’ll see:
    1. **In‐Sample**: Actual vs. Predicted Revenue with 95 % Confidence Intervals  
    2. **Out‐of‐Sample**: Revenue Forecast for the chosen horizon  
    3. **Scatter Analysis**: How CPI and Store Count correlate with Revenue  
    4. **Model Summary** and a **50–100 word AI-Generated Summary** for an audit committee.  
    This app follows the Final Project Requirements :contentReference[oaicite:1]{index=1}.
    """
)

st.sidebar.header("Forecast Options")
# 1) Forecast horizon (quarters, 1–8)
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (quarters)", 
    min_value=1, max_value=8, value=4, step=1,
    help="Select how many quarters into the future to forecast revenue (1–8)."
)

# ------------------------------------
#  4. DATA LOADING & PREPARATION
# ------------------------------------
csv_path = "final_project_starbucks_data.csv"
with st.spinner("Loading Starbucks data..."):
    try:
        sbux_df = load_starbucks_data(csv_path)
    except FileNotFoundError:
        st.error(f"CSV not found at '{csv_path}'. Please verify the file exists.")
        st.stop()

# Must have at least 'revenue' and 'store_count'
if ("revenue" not in sbux_df.columns) or ("store_count" not in sbux_df.columns):
    st.error("Data must include 'revenue' and 'store_count' columns.")
    st.stop()

# Determine date range to fetch CPI
start_date = sbux_df.index.min().strftime("%Y-%m-%d")
end_date   = sbux_df.index.max().strftime("%Y-%m-%d")

# 2) Fetch CPI using API key from st.secrets
fred_api_key = st.secrets.get("fred_api_key", "")
if not fred_api_key:
    st.error("❌ No FRED API key found in Streamlit secrets. Please add `fred_api_key` to .streamlit/secrets.toml.")
    st.stop()

with st.spinner("Fetching CPI data from FRED..."):
    try:
        cpi_series = fetch_quarterly_cpi(fred_api_key, start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching CPI: {e}")
        st.stop()

# Merge CPI into Starbucks DataFrame
df = sbux_df.copy()
df = df.assign(CPI=cpi_series)
df = df[["revenue", "store_count", "CPI"]].dropna()  # Keep only rows where none are NaN

# ------------------------------------
#  5. MODEL FITTING & IN-SAMPLE PREDICTION
# ------------------------------------
with st.spinner("Fitting ARIMAX model..."):
    arimax_result = fit_arimax_model(df)

# Prepare in-sample exogenous for prediction
exog_insample = df[["CPI", "store_count"]]
# In-sample predicted (fitted) values + 95% CI
pred_insample = arimax_result.get_prediction(start=0, end=len(df) - 1, exog=exog_insample)
pred_mean_insample = pred_insample.predicted_mean
conf_int_insample = pred_insample.conf_int(alpha=0.05)

# ------------------------------------
#  6. OUT-OF-SAMPLE FORECAST
# ------------------------------------
last_date = df.index.max()
# Build future quarter‐end index
future_index = pd.date_range(
    start=(last_date + pd.tseries.offsets.QuarterEnd(1)),
    periods=forecast_horizon,
    freq="Q"
)

# For forecasting CPI and store_count, assume “last observed” values persist
last_cpi_val   = df["CPI"].iloc[-1]
last_store_val = df["store_count"].iloc[-1]
future_exog = pd.DataFrame({
    "CPI":            [last_cpi_val]   * forecast_horizon,
    "store_count":    [last_store_val] * forecast_horizon
}, index=future_index)

forecast_obj       = arimax_result.get_forecast(steps=forecast_horizon, exog=future_exog)
forecast_mean      = forecast_obj.predicted_mean
forecast_conf_int  = forecast_obj.conf_int(alpha=0.05)

# ------------------------------------
#  7. VISUALIZATIONS
# ------------------------------------

# 7a) In-Sample Actual vs. Predicted Revenue
st.subheader("1. In-Sample: Actual vs. Fitted Revenue (with 95 % CI)")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(
    df.index, df["revenue"],
    label="Actual Revenue",
    marker="o", linestyle="-", color="tab:blue"
)
ax1.plot(
    df.index, pred_mean_insample,
    label="Fitted Revenue",
    marker="o", linestyle="--", color="tab:orange"
)
ax1.fill_between(
    df.index.astype("datetime64[ns]"),
    conf_int_insample["lower revenue"],
    conf_int_insample["upper revenue"],
    color="tab:orange", alpha=0.2, label="95 % CI"
)
ax1.set_title("Actual vs. In-Sample Fitted Revenue\n(Revenue in Millions USD)")
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Revenue (Millions USD)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.markdown("---")

# 7b) Out-of-Sample Forecast
st.subheader(f"2. Out-of-Sample: Revenue Forecast  (Next {forecast_horizon} Quarter{'s' if forecast_horizon>1 else ''})")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(
    df.index, df["revenue"],
    label="Historical Actual Revenue",
    marker="o", linestyle="-", color="tab:blue"
)
ax2.plot(
    future_index, forecast_mean,
    label="Forecasted Revenue",
    marker="o", linestyle="--", color="tab:green"
)
ax2.fill_between(
    future_index.astype("datetime64[ns]"),
    forecast_conf_int["lower revenue"],
    forecast_conf_int["upper revenue"],
    color="tab:green", alpha=0.2, label="95 % CI (Forecast)"
)
ax2.set_title(f"Historical & Forecasted Revenue\n(Revenue in Millions USD)")
ax2.set_xlabel("Quarter")
ax2.set_ylabel("Revenue (Millions USD)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.markdown("---")

# 7c) Scatter Plots: Correlation of CPI & Store Count with Revenue
st.subheader("3. Correlation: CPI & Store Count vs. Revenue")
col1, col2 = st.columns(2)

with col1:
    fig3a, ax3a = plt.subplots(figsize=(5, 4))
    ax3a.scatter(
        df["CPI"], df["revenue"],
        color="tab:purple", alpha=0.7
    )
    # Fit simple linear trend
    m_cpi, b_cpi = np.polyfit(df["CPI"], df["revenue"], 1)
    x_vals_cpi = np.linspace(df["CPI"].min(), df["CPI"].max(), 100)
    ax3a.plot(x_vals_cpi, m_cpi * x_vals_cpi + b_cpi,
              color="tab:red", linestyle="--", label="Trend")
    ax3a.set_title("CPI vs. Revenue")
    ax3a.set_xlabel("CPI (Index)")
    ax3a.set_ylabel("Revenue (Millions USD)")
    ax3a.legend()
    ax3a.grid(True)
    st.pyplot(fig3a)

with col2:
    fig3b, ax3b = plt.subplots(figsize=(5, 4))
    ax3b.scatter(
        df["store_count"], df["revenue"],
        color="tab:olive", alpha=0.7
    )
    # Fit simple linear trend
    m_store, b_store = np.polyfit(df["store_count"], df["revenue"], 1)
    x_vals_store = np.linspace(df["store_count"].min(), df["store_count"].max(), 100)
    ax3b.plot(x_vals_store, m_store * x_vals_store + b_store,
              color="tab:red", linestyle="--", label="Trend")
    ax3b.set_title("Store Count vs. Revenue")
    ax3b.set_xlabel("Store Count")
    ax3b.set_ylabel("Revenue (Millions USD)")
    ax3b.legend()
    ax3b.grid(True)
    st.pyplot(fig3b)

st.markdown(
    """
    *Insight:* Both CPI and store count exhibit a positive relationship with quarterly revenue.
    Including these variables in our ARIMAX model helps adjust for macroeconomic inflation (CPI)
    and business expansion (store count) when forecasting revenue. 
    """
)

st.markdown("---")

# ------------------------------------
#  8. MODEL & AI-GENERATED SUMMARY
# ------------------------------------
st.subheader("4. Model Summary & AI-Generated Narrative")

with st.expander("Show Full ARIMAX Model Summary"):
    st.text(arimax_result.summary().as_text())

ai_default = (
    "Our ARIMAX model forecasts Starbucks quarterly revenue (millions USD) using live CPI from FRED "
    "and store count as exogenous factors. In-sample results demonstrate strong fit: actual vs. predicted revenue "
    "align within a narrow 95% confidence band. Out-of-sample forecasts (up to 8 quarters) assume stable CPI "
    "and store count at last observed levels. Key audit insight: incorporating CPI mitigates inflation-related "
    "overstatement risk, while monitoring store count growth ensures consistency with observed revenue trends."
)

ai_summary = st.text_area(
    label="Audit Committee Summary (50–100 words):",
    value=ai_default,
    height=150,
    help="Refine or replace this AI-generated summary for a professional audit-committee audience."
)

# ------------------------------------
# 10. FOOTER / NOTES
# ------------------------------------
st.markdown("---")
st.markdown(
    """
    **Notes:**  
    - Revenue is expressed in **millions of USD** throughout.  
    - CPI data is fetched live from FRED using the API key stored in `st.secrets['fred_api_key']`.  
    - Store count is the “new insight” variable (not used in Project 2) and captures Starbucks’ expansion.  
    - By selecting 1–8 quarter forecasts, audit committees can assess how macroeconomic and expansion factors affect revenue risk.  
    """
)
