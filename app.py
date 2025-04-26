import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional   

# Configure app
APP_TITLE = "Financial Time-Series Analysis Dashboard"
APP_ICON = "📈"
CACHE_TTL = 60 * 60  # 1 hour
MIN_INVEST = 100  # $ minimum allowed

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard")

# Import custom modules
from data.data_loader import AlphaVantageAPI, filter_data_by_timeframe
from data.data_processor import DataProcessor
from my_analysis.decomposition import decompose_time_series, check_time_series_properties
from my_analysis.statistical_tests import adf_test_summary, kpss_test_summary, suggest_arima_orders
from my_analysis.indicators.moving_averages import simple_moving_average, exponential_moving_average
from my_analysis.indicators.oscillators import relative_strength_index, stochastic_oscillator
from my_analysis.indicators.volume import on_balance_volume, volume_weighted_average_price
from models.arima_model import ARIMAModel, auto_arima_model
from models.lstm_model import LSTMModel, create_lstm_model
from models.linear_models import DLinearModel, NLinearModel
from models.timegpt_model import TimeGPTModel, create_timegpt_model
from visualization.price_plots import plot_ohlc, plot_forecast, plot_decomposition, plot_acf_pacf
from visualization.performance_plots import plot_model_comparison, plot_equity_curve, plot_drawdowns
from statsmodels.tsa.stattools import acf, pacf

# Helper function for percentage formatting
def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"

# Simple timer context manager
from contextlib import contextmanager
@contextmanager
def timed(msg: str):
    start = datetime.now()
    yield
    dur = (datetime.now() - start).total_seconds()
    logger.debug(f"{msg} finished in {dur:.2f}s")

# Define constants
TIME_PERIODS = ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"]
ANALYSIS_TABS = [
    "Basic Analysis",
    "Technical Indicators",
    "Time Series Decomposition",
    "Statistical Tests",
    "Forecasting Models",
    "Model Comparison",
]

@dataclass
class UserInputs:
    security_type : str
    ticker        : str
    time_period   : str
    api_key       : str
    timegpt_key   : str
    investment    : float
    analysis_tab  : str
    # new optional fields (only filled for the two forecasting tabs)
    horizon       : Optional[int]   = None
    split_ratio   : Optional[float] = None
    model_choice  : Optional[str]   = None

# Metrics calculation helper
def _metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    """Quick calc of common regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    return dict(rmse=rmse, mae=mae, mape=mape, r2=r2)



# Sidebar renderer
def render_sidebar() -> UserInputs:
    """Draw sidebar widgets and return one bundle of user inputs."""
    st.sidebar.header("Input Parameters")

    # ---------- basic inputs -------------------------------------------------
    security_type = st.sidebar.selectbox("Security Type", ["Stocks", "Forex", "Crypto"])

    if security_type == "Stocks":
        ticker = st.sidebar.text_input("Stock Symbol", "AAPL").upper().strip()
    elif security_type == "Forex":
        c1, c2 = st.sidebar.columns(2)
        ticker = f"{c1.text_input('From', 'USD').upper().strip()}/{c2.text_input('To', 'EUR').upper().strip()}"
    else:                               # Crypto
        c1, c2 = st.sidebar.columns(2)
        ticker = f"{c1.text_input('Crypto', 'BTC').upper().strip()}/{c2.text_input('Market', 'USD').upper().strip()}"

    time_period = st.sidebar.selectbox("Time Period", TIME_PERIODS)
    api_key      = st.sidebar.text_input("Alpha Vantage API Key", "", type="password")
    timegpt_key  = st.sidebar.text_input("TimeGPT API Key (optional)", "", type="password")
    investment   = st.sidebar.number_input("Initial Investment Amount ($)", min_value=MIN_INVEST, value=10_000)

    analysis_tab = st.sidebar.radio("Analysis Type", ANALYSIS_TABS)

    # ---------- extra widgets only for forecasting tabs ----------------------
    horizon = split_ratio = model_choice = None

    if analysis_tab == "Forecasting Models":
        st.sidebar.markdown("### Forecast settings")
        horizon      = st.sidebar.slider("Forecast horizon (days)", 1, 90, 30)
        split_ratio  = st.sidebar.slider("Train/Test split", 0.5, 0.9, 0.8, 0.05)
        model_choice = st.sidebar.selectbox("Choose model",
                                            ["ARIMA", "LSTM", "D-Linear", "N-Linear", "TimeGPT"])

    elif analysis_tab == "Model Comparison":        # we’ll compare ALL models
        st.sidebar.markdown("### Comparison settings")
        horizon     = st.sidebar.slider("Forecast horizon (days)", 1, 90, 30)
        split_ratio = st.sidebar.slider("Train/Test split", 0.5, 0.9, 0.8, 0.05)

    # ------------------------------------------------------------------------
    return UserInputs(security_type, ticker, time_period,
                      api_key, timegpt_key, investment, analysis_tab,
                      horizon, split_ratio, model_choice)


# Data fetching function
@st.cache_data(ttl=CACHE_TTL, show_spinner="📡 Contacting Alpha Vantage…")
def fetch_data(security_type: str, ticker: str, api_key: str, period: str) -> pd.DataFrame | None:
    """
    Hit Alpha Vantage once, cache for an hour.
    Returns a *clean* DataFrame indexed by datetime (or None on failure).
    """
    if not api_key:
        st.error("Please enter your Alpha Vantage API key in the sidebar.")
        return None

    api = AlphaVantageAPI(api_key)

    with timed(f"fetch {ticker}"):
        try:
            if security_type == "Stocks":
                size = "compact" if period == "1 Month" else "full"
                df, err = api.fetch_time_series_daily(ticker, size)

            elif security_type == "Forex":
                base, quote = ticker.split("/")
                df, err = api.fetch_forex(base, quote, interval="daily")

            else:  # Crypto
                base, quote = ticker.split("/")
                df, err = api.fetch_crypto(base, quote, interval="daily")

            if err:
                st.error(f"Alpha Vantage error: {err}")
                return None
        except Exception as e:
            st.error(f"Request failed: {e}")
            return None

    # Filter & basic sanity-checks
    df = filter_data_by_timeframe(df, period)

    if df is None or df.empty:
        st.warning("No data returned for that combination — try a longer time period?")
        return None

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    logger.info(f"Fetched {len(df):,} rows for {ticker}")
    return df

# Basic Analysis tab
def render_basic(df: pd.DataFrame, ticker: str) -> None:
    st.subheader("Price Chart")
    st.plotly_chart(
        plot_ohlc(df, title=f"{ticker} – OHLC Chart", volume=True,
                  ma_periods=[20, 50], return_fig=True),
        use_container_width=True
    )

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${df.close.iloc[-1]:.2f}")
    col2.metric("Avg Volume", f"{df.volume.mean():,.0f}" if "volume" in df.columns else "–")

    price_change = df.close.iloc[-1] - df.close.iloc[0]
    price_pct = price_change / df.close.iloc[0] * 100
    col3.metric("Price Change", f"${price_change:.2f}", pct(price_pct))

    vol = df.close.pct_change().std() * 100
    col4.metric("Volatility (σ)", pct(vol))

    # Returns distribution
    st.subheader("Returns Distribution")
    returns = df.close.pct_change().dropna() * 100
    fig = go.Figure(go.Histogram(x=returns, nbinsx=30, marker_color="lightblue"))
    fig.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df)

# Technical Indicators tab
def render_technicals(df: pd.DataFrame) -> None:
    # Moving averages
    st.subheader("Moving Averages")
    periods = [10, 20, 50, 200]
    df_ma = exponential_moving_average(
        simple_moving_average(df, "close", periods),
        "close", periods
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ma.index, y=df_ma.close,
                             name="Close", line=dict(color="black", width=2)))
    for p in periods:
        fig.add_trace(go.Scatter(x=df_ma.index, y=df_ma[f"sma_{p}"],
                                 name=f"SMA {p}", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df_ma.index, y=df_ma[f"ema_{p}"],
                                 name=f"EMA {p}"))
    fig.update_layout(legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("Relative Strength Index (RSI 14)")
    df_rsi = relative_strength_index(df, "close", 14)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Price", "RSI"),
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df_rsi.index, y=df_rsi.close,
                             name="Close", line=dict(color="black", width=2)),
                  1, 1)
    fig.add_trace(go.Scatter(x=df_rsi.index, y=df_rsi.rsi, name="RSI",
                             line=dict(color="blue")), 2, 1)
    fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    if "volume" in df.columns:
        st.subheader("Volume Analysis")
        df_vol = volume_weighted_average_price(on_balance_volume(df), 20)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=("Price", "Volume", "OBV"))
        fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol.close,
                                 name="Close", line=dict(color="black")), 1, 1)
        fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol.vwap,
                                 name="VWAP 20", line=dict(color="purple")), 1, 1)

        colors = ["green" if c >= o else "red"
                  for o, c in zip(df_vol.open, df_vol.close)]
        fig.add_trace(go.Bar(x=df_vol.index, y=df_vol.volume,
                             marker_color=colors, name="Vol"), 2, 1)
        fig.add_trace(go.Scatter(x=df_vol.index, y=df_vol.obv,
                                 name="OBV", line=dict(color="blue")), 3, 1)
        st.plotly_chart(fig, use_container_width=True)


# Time Series Decomposition tab
def render_decomposition(df: pd.DataFrame, ticker: str) -> None:
    st.subheader("Time-series Decomposition")

    col1, col2 = st.columns(2)
    model = col1.selectbox("Model", ["additive", "multiplicative"])
    auto_p = col2.checkbox("Auto-detect period", True)
    period = None if auto_p else col2.slider("Period", 2, 365, 252)

    try:
        decomp = decompose_time_series(df.close, model=model, period=period)
    except Exception as e:
        st.error(f"Decomposition failed – {e}")
        return
    st.plotly_chart(
        plot_decomposition(decomp, f"{ticker} – Decomposition", return_fig=True),
        use_container_width=True
    )

    if auto_p:
        st.info(f"Detected seasonality period ≈ {decomp['period']}")

    # Property summary
    props = check_time_series_properties(df.close)
    st.write("**Stationarity:**",
             "Stationary" if props["stationarity"]["is_stationary"] else "Non-stationary")
    st.write("ADF p-value:", f"{props['stationarity']['adf_test']['p_value']:.4f}")
    st.write("KPSS p-value:", f"{props['stationarity']['kpss_test']['p_value']:.4f}")

    # Rolling stats
    st.subheader("Rolling Statistics")
    win = st.slider("Window", 5, 252, 20)
    roll = DataProcessor.rolling_statistics(df.close, win)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.close, name="Close", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.rolling_mean,
                             name=f"Mean {win}", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.rolling_mean + 2*roll.rolling_std,
                             name="+2σ", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.rolling_mean - 2*roll.rolling_std,
                             name="-2σ", line=dict(color="red", dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

# Statistical Tests tab
def render_stats_tests(df: pd.DataFrame) -> None:
    st.subheader("Augmented Dickey-Fuller")
    adf = adf_test_summary(df.close)
    st.code(adf["summary"])

    st.subheader("KPSS Test")
    kpss = kpss_test_summary(df.close)
    st.code(kpss["summary"])

    # ACF / PACF plots ---------------------------------------------------------
    st.subheader("ACF & PACF")
    use_diff = st.checkbox("Use differenced series",
                           value=not adf["stationary"])
    series = df.close.diff().dropna() if use_diff else df.close

    # --- START PATCH ----------------------------------------------------------
    # statsmodels requires nlags ≤ 50 % of sample size
    max_lags = max(3, (len(series) // 2) - 1)        # at least 3
    nlags = st.slider("Lags", 1, max_lags,
                      min(20, max_lags))              # sensible default

    try:
        acf_vals, acf_ci  = acf(series, nlags, alpha=0.05, fft=True)
        pacf_vals, pacf_ci = pacf(series, nlags, alpha=0.05)
    except ValueError as e:                           # too many lags for sample
        st.warning(str(e))
        return
    # --- END PATCH ------------------------------------------------------------

    lags = np.arange(len(acf_vals))

    st.plotly_chart(
        plot_acf_pacf(acf_vals, pacf_vals, lags,
                      acf_ci, pacf_ci,
                      title=f"ACF & PACF ({'diff' if use_diff else 'orig'})",
                      return_fig=True),
        use_container_width=True
    )

    st.subheader("Suggested ARIMA Orders")
    suggestion = suggest_arima_orders(df.close)
    st.code(suggestion["summary"])


# Analysis tab dispatcher
def render_analysis_tab(tab: str, df: pd.DataFrame, ticker: str) -> None:
    """Call the right renderer based on radio selection."""
    if tab == "Basic Analysis":
        render_basic(df, ticker)
    elif tab == "Technical Indicators":
        render_technicals(df)
    elif tab == "Time Series Decomposition":
        render_decomposition(df, ticker)
    elif tab == "Statistical Tests":
        render_stats_tests(df)
    else:
        st.warning("This tab belongs to the forecasting / model section.")


# Forecasting Models tab
# ── TAB 5: Forecasting Models ─────────────────────────────────────────
def render_forecasting(
        df: pd.DataFrame,
        horizon: int,
        model_type: str,
        split_ratio: float,
        timegpt_key: str
) -> tuple[pd.Series | None, dict]:
    """
    Train the selected model, make a forecast and return (forecast, metrics).
    The function is now guard-railed so horizon never exceeds the length
    of the test-set, preventing the “inconsistent numbers of samples” error.
    """
    # ------------------------------------------------------------------
    split = int(len(df) * split_ratio)
    train, test = df.close.iloc[:split], df.close.iloc[split:]

    # never ask for more steps than we actually have
    horizon = min(horizon, len(test))
    # ------------------------------------------------------------------

    # ――― model-specific training / prediction ――――――――――――――――――――――
    if model_type == "ARIMA":
        auto = st.checkbox("Use auto_arima", value=True)
        if auto:
            res = auto_arima_model(
                train, max_p=5, max_d=2, max_q=5,
                seasonal=False, information_criterion="aic"
            )
            if "error" in res:
                st.error(res["error"])
                return None, {}
            model = res["arima_model"]
        else:
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)
            model = ARIMAModel(order=(p, d, q))
            model.fit(train)

        fc = model.predict(steps=horizon)["forecast"]

    elif model_type == "LSTM":
        seq = st.number_input("Sequence length", 1, 100, 10)
        layers = [64, 32]
        res = create_lstm_model(
            train, sequence_length=seq, layers=layers,
            dropout_rate=0.2, epochs=100,
            batch_size=32, patience=10
        )
        if "error" in res:
            st.error(res["error"])
            return None, {}
        model = res["model"]
        fc = model.predict(steps=horizon)["forecast"]

    elif model_type == "D-Linear":
        model = DLinearModel(
            sequence_length=24, horizon=min(horizon, 30),
            epochs=100, batch_size=32, patience=10
        )
        if model.fit(train).get("error"):
            st.error("D-Linear fit failed")
            return None, {}
        fc = model.predict(steps=horizon)["forecast"]

    elif model_type == "N-Linear":
        model = NLinearModel(
            sequence_length=24, horizon=min(horizon, 30),
            epochs=100, batch_size=32, patience=10
        )
        if model.fit(train).get("error"):
            st.error("N-Linear fit failed")
            return None, {}
        fc = model.predict(steps=horizon)["forecast"]

    elif model_type == "TimeGPT":
        if not timegpt_key:
            st.warning("TimeGPT key missing.")
            return None, {}
        res = create_timegpt_model(
            train, api_key=timegpt_key,
            freq="auto", finetune_steps=10, finetune=True
        )
        if "error" in res:
            st.error(res["error"])
            return None, {}
        model = res["model"]
        fc = model.predict(steps=horizon)["forecast"]

    else:
        st.error("Unknown model")
        return None, {}

    # ------------------------------------------------------------------
    # Ensure forecast length == horizon (some wrappers return longer)
    fc = fc.iloc[:horizon]
    # ------------------------------------------------------------------

    # ――― metrics & chart ――――――――――――――――――――――――――――――――――――――――――
    metrics = _metrics(test.iloc[:horizon], fc)

    fig = plot_forecast(
        train,
        fc,
        test.iloc[:horizon],            # align lengths for plotting
        f"{model_type} Forecast",
        y_label="Price",
        return_fig=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{metrics['rmse']:.4f}")
    col2.metric("MAE",  f"{metrics['mae']:.4f}")
    col3.metric("MAPE", pct(metrics['mape']))
    col4.metric("R²",   f"{metrics['r2']:.4f}")

    return fc, metrics


# Model Comparison tab
def render_model_comparison(df: pd.DataFrame, horizon: int, split_ratio: float,
                            chosen: list[str], invest: float, timegpt_key: str):
    if not chosen:
        st.info("Select at least one model on the left.")
        return

    forecasts, metrics = {}, {}
    for m in chosen:
        st.write(f"### {m}")
        fc, met = render_forecasting(df, horizon, m, split_ratio, timegpt_key)
        if fc is not None:
            forecasts[m] = fc
            metrics[m]   = met
        st.divider()

    # Metrics comparison chart
    if metrics:
        st.subheader("Performance Comparison (lower RMSE better)")
        fig = plot_model_comparison(metrics, "rmse", "Model Comparison", return_fig=True)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        metric_df = pd.DataFrame(metrics).T.round(4).sort_values("rmse")
        st.table(metric_df)

    # Simple investment simulation
    if invest and forecasts:
        st.subheader("Buy-&-Hold vs Signal Strategy")
        returns = {}
        test = df.close.iloc[int(len(df)*split_ratio):]

        for name, fc in forecasts.items():
            sig = (fc.shift(1) > fc)  # sell signal
            act_r = test.pct_change().dropna()
            strat = act_r.copy(); strat[sig] = 0
            returns[name] = strat*100

        returns["Buy & Hold"] = test.pct_change().dropna()*100
        st.plotly_chart(plot_equity_curve(invest, returns,
                                          "Equity Curve", True),
                        use_container_width=True)
        st.plotly_chart(plot_drawdowns(returns, "Drawdowns", True),
                        use_container_width=True)



def main() -> None:
    """Top-level dispatcher – shows sidebar, fetches data, routes to the
    correct analysis/forecast tab."""
    ui = render_sidebar()

    if not st.sidebar.button("Analyze"):
        return

    # ── 1. Download data (cached) ──────────────────────────────────────
    data = fetch_data(
        ui.security_type,
        ui.ticker,
        ui.api_key,
        ui.time_period,
    )
    if data is None:
        st.stop()

    # ── 2. Classic analysis tabs ───────────────────────────────────────
    if ui.analysis_tab in (
        "Basic Analysis",
        "Technical Indicators",
        "Time Series Decomposition",
        "Statistical Tests",
    ):
        render_analysis_tab(ui.analysis_tab, data, ui.ticker)
        return

    # 3️⃣  Single-model forecasting

    if ui.analysis_tab == "Forecasting Models":
        if ui.horizon is None or ui.split_ratio is None or ui.model_choice is None:
            st.warning("Please set horizon, split and model in the sidebar.")
        else:
            render_forecasting(
                data,
                horizon     = ui.horizon,
                model_type  = ui.model_choice,
                split_ratio = ui.split_ratio,
                timegpt_key = ui.timegpt_key,
            )


    # 4️⃣  Full model comparison (always runs all five models)
    elif ui.analysis_tab == "Model Comparison":
        all_models = ["ARIMA", "LSTM", "D-Linear", "N-Linear", "TimeGPT"]
        render_model_comparison(data,
                                horizon      = ui.horizon,
                                split_ratio  = ui.split_ratio,
                                chosen       = all_models,
                                invest       = ui.investment,
                                timegpt_key  = ui.timegpt_key)    


# ── Run the app ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

