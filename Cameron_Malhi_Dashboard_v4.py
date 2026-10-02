import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ============================================================
# PAGE / BRANDING
# ============================================================
st.set_page_config(
    page_title="US HY Cross-Asset RV Monitor",
    page_icon="📈",
    layout="wide",
)

BG = "#0b0f14"
PANEL = "#111820"
BORDER = "#263442"
TEXT = "#e8eef5"
MUTED = "#92a4b5"
GRID = "#20303d"
BLUE = "#67a4ff"
ORANGE = "#ff7a59"
GREEN = "#29d3a2"
PURPLE = "#b07cff"
YELLOW = "#f3c969"
RED = "#ff6b6b"

st.markdown(
    f"""
    <style>
        .stApp {{ background:{BG}; color:{TEXT}; }}
        [data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {BORDER}; }}
        [data-testid="stMetric"] {{
            background:{PANEL}; border:1px solid {BORDER}; border-radius:12px;
            padding:14px 16px;
        }}
        [data-testid="stMetricLabel"] {{ color:#9fc8ff !important; }}
        [data-testid="stMetricValue"] {{ color:{TEXT} !important; }}
        [data-testid="stMetricDelta"] {{ color:#b8c8d8 !important; }}
        .rv-banner {{
            border:1px solid {BORDER}; border-radius:14px; padding:18px 20px;
            margin:8px 0 14px; background:{PANEL};
        }}
        .rv-kicker {{ font-size:.78rem; color:#9fc8ff; text-transform:uppercase; letter-spacing:.10em; }}
        .rv-head {{ font-size:1.45rem; font-weight:750; margin-top:5px; color:{TEXT}; }}
        .rv-copy {{ color:#d1dbe5; margin-top:4px; }}
        .rv-meta {{ color:{MUTED}; margin-top:9px; font-size:.88rem; }}
        .small-note {{ color:{MUTED}; font-size:.88rem; }}
        .driver-box {{
            background:{PANEL}; border:1px solid {BORDER}; border-radius:12px;
            padding:14px 16px; margin-bottom:12px;
        }}
        div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:10px; }}
        hr {{ border-color:{BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA SOURCES
# ============================================================
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED = {
    "hy_oas": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS (%)
    "ust10": "DGS10",           # US 10Y Treasury yield (%)
    "fed_funds": "DFF",         # Effective Fed Funds rate (%)
}
YAHOO = {
    "spx": "^GSPC",
    "vix": "^VIX",
    "oil": "CL=F",
    "dxy": "DX-Y.NYB",
}

# Long history for backtesting even if the display is shorter.
ANALYSIS_HISTORY_YEARS = 15

# ============================================================
# HELPERS
# ============================================================
def last_value(series: pd.Series):
    s = series.dropna()
    return np.nan if s.empty else float(s.iloc[-1])


def rolling_zscore(series: pd.Series, window: int):
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std


def percentile_rank(series: pd.Series, value: float):
    s = series.dropna()
    if s.empty or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)


def fmt_num(x, pattern=".1f", suffix=""):
    return "n/a" if pd.isna(x) else f"{x:{pattern}}{suffix}"


def style_figure(fig: go.Figure, height=None, legend=True):
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Arial"),
        hoverlabel=dict(bgcolor=PANEL, font_color=TEXT),
        margin=dict(l=12, r=12, t=55, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.10) if legend else dict(visible=False),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    return fig


@st.cache_data(ttl=3600)
def load_fred(series_id: str) -> pd.Series:
    df = pd.read_csv(FRED_URL.format(series_id), index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s


@st.cache_data(ttl=3600)
def load_yahoo(ticker: str, years: int) -> pd.Series:
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years + 1)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def try_optional_yahoo(ticker: str, years: int):
    try:
        return load_yahoo(ticker, years), None
    except Exception as exc:
        return pd.Series(dtype=float), f"{ticker}: {type(exc).__name__}"


def build_daily(history_years: int):
    # Core series: app stops if any of these fail.
    hy = load_fred(FRED["hy_oas"]) * 100
    ust10 = load_fred(FRED["ust10"])
    fed_funds = load_fred(FRED["fed_funds"])
    spx = load_yahoo(YAHOO["spx"], history_years)
    vix = load_yahoo(YAHOO["vix"], history_years)

    oil, oil_error = try_optional_yahoo(YAHOO["oil"], history_years)
    dxy, dxy_error = try_optional_yahoo(YAHOO["dxy"], history_years)

    source_last = {
        "HY OAS": hy.index.max(),
        "US 10Y": ust10.index.max(),
        "Fed Funds": fed_funds.index.max(),
        "S&P 500": spx.index.max(),
        "VIX": vix.index.max(),
        "WTI": oil.index.max() if not oil.empty else pd.NaT,
        "DXY": dxy.index.max() if not dxy.empty else pd.NaT,
    }

    df = pd.concat(
        {
            "hy_oas": hy,
            "ust10": ust10,
            "fed_funds": fed_funds,
            "spx": spx,
            "vix": vix,
            "oil": oil,
            "dxy": dxy,
        },
        axis=1,
        sort=False,
    ).sort_index()

    # Forward-fill only after preserving source freshness metadata.
    df = df.ffill()
    df = df.dropna(subset=["hy_oas", "spx", "vix", "ust10"])
    cutoff = df.index.max() - pd.DateOffset(years=history_years)
    df = df[df.index >= cutoff].copy()

    warnings = [x for x in [oil_error, dxy_error] if x]
    return df, source_last, warnings


# ============================================================
# DAILY MARKET STRESS
# ============================================================
def add_daily_signals(df: pd.DataFrame, z_days: int):
    out = df.copy()
    out["z_credit"] = rolling_zscore(np.log(out["hy_oas"]), z_days)
    out["z_equity"] = -rolling_zscore(np.log(out["spx"]), z_days)
    out["stress_gap"] = out["z_credit"] - out["z_equity"]
    out["spx_3m_ret"] = out["spx"].pct_change(63)
    out["hy_3m_change"] = out["hy_oas"].diff(63)
    return out


# ============================================================
# WALK-FORWARD MULTI-FACTOR MODEL
# ΔHY OAS = a + b1(SPX ret) + b2(ΔVIX) + b3(Δ10Y) + residual
# ============================================================
def build_weekly_model(df: pd.DataFrame, model_window: int, resid_window: int):
    wk = df[["hy_oas", "spx", "vix", "ust10", "oil", "dxy", "fed_funds"]].resample("W-FRI").last()

    wk["d_hy"] = wk["hy_oas"].diff()
    wk["spx_ret"] = np.log(wk["spx"]).diff()
    wk["d_vix"] = wk["vix"].diff()
    wk["d_ust10"] = wk["ust10"].diff()  # percentage-point change

    model_cols = [
        "pred_d_hy", "resid", "alpha", "beta_spx", "beta_vix", "beta_ust10",
        "contrib_intercept", "contrib_spx", "contrib_vix", "contrib_ust10", "train_r2"
    ]
    for c in model_cols:
        wk[c] = np.nan

    features = ["spx_ret", "d_vix", "d_ust10"]

    for i in range(model_window, len(wk)):
        # Strictly prior data: current week is never used to fit its own coefficients.
        train = wk.iloc[i - model_window:i].dropna(subset=["d_hy"] + features)
        current = wk.iloc[i]

        if len(train) < max(40, model_window // 2) or current[features].isna().any():
            continue

        X = np.column_stack([
            np.ones(len(train)),
            train["spx_ret"].values,
            train["d_vix"].values,
            train["d_ust10"].values,
        ])
        y = train["d_hy"].values

        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, b_spx, b_vix, b_ust = coef

        c_intercept = float(alpha)
        c_spx = float(b_spx * current["spx_ret"])
        c_vix = float(b_vix * current["d_vix"])
        c_ust = float(b_ust * current["d_ust10"])
        pred = c_intercept + c_spx + c_vix + c_ust
        resid = float(current["d_hy"] - pred)

        yhat_train = X @ coef
        ss_res = np.sum((y - yhat_train) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        train_r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        idx = wk.index[i]
        wk.loc[idx, model_cols] = [
            pred, resid, alpha, b_spx, b_vix, b_ust,
            c_intercept, c_spx, c_vix, c_ust, train_r2
        ]

    wk["resid_z"] = rolling_zscore(wk["resid"], resid_window)
    return wk


# ============================================================
# ROBUST EVENT DETECTION + BACKTEST
# ============================================================
def backtest_events(wk: pd.DataFrame, threshold: float, cooldown: int = 4):
    valid = wk.dropna(subset=["resid_z"]).copy()
    if valid.empty:
        return pd.DataFrame()

    z = valid["resid_z"]
    state = pd.Series(
        np.select([z >= threshold, z <= -threshold], [1, -1], default=0),
        index=valid.index,
        dtype=int,
    )

    # Fixes the old bug: the first valid observation can now trigger an event.
    prev_state = state.shift(1).fillna(0).astype(int)
    is_new_event = (state != 0) & (state != prev_state)
    candidates = list(valid.index[is_new_event])

    accepted = []
    last_pos = -10_000
    for date in candidates:
        pos = wk.index.get_loc(date)
        if pos - last_pos >= cooldown:
            accepted.append(date)
            last_pos = pos

    rows = []
    for date in accepted:
        i = wk.index.get_loc(date)
        sig = float(wk.loc[date, "resid_z"])
        side = "Credit wide" if sig > 0 else "Credit tight"

        row = {
            "date": date,
            "signal": side,
            "resid_z": sig,
            "hy_oas_entry": float(wk.loc[date, "hy_oas"]),
            "spx_entry": float(wk.loc[date, "spx"]),
        }

        for horizon in (4, 12):
            if i + horizon < len(wk):
                future = wk.iloc[i + horizon]
                oas_move = float(future["hy_oas"] - wk.iloc[i]["hy_oas"])
                spx_ret = float(future["spx"] / wk.iloc[i]["spx"] - 1)
                mean_reverted = (
                    (sig > 0 and oas_move < 0) or
                    (sig < 0 and oas_move > 0)
                )
                row[f"oas_{horizon}w"] = oas_move
                row[f"spx_{horizon}w"] = spx_ret
                row[f"mr_{horizon}w"] = bool(mean_reverted)
            else:
                row[f"oas_{horizon}w"] = np.nan
                row[f"spx_{horizon}w"] = np.nan
                row[f"mr_{horizon}w"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def backtest_by_side(events: pd.DataFrame, horizon: int):
    rows = []
    for side in ["Credit wide", "Credit tight"]:
        if events.empty:
            subset = pd.DataFrame()
        else:
            subset = events[events["signal"] == side].dropna(
                subset=[f"oas_{horizon}w", f"spx_{horizon}w", f"mr_{horizon}w"]
            )

        if subset.empty:
            rows.append({
                "Signal": side,
                "N": 0,
                "Hit rate": np.nan,
                "Avg OAS move": np.nan,
                "Median OAS move": np.nan,
                "Avg S&P return": np.nan,
                "Median S&P return": np.nan,
            })
        else:
            rows.append({
                "Signal": side,
                "N": len(subset),
                "Hit rate": float(subset[f"mr_{horizon}w"].mean()),
                "Avg OAS move": float(subset[f"oas_{horizon}w"].mean()),
                "Median OAS move": float(subset[f"oas_{horizon}w"].median()),
                "Avg S&P return": float(subset[f"spx_{horizon}w"].mean()),
                "Median S&P return": float(subset[f"spx_{horizon}w"].median()),
            })
    return pd.DataFrame(rows)


# ============================================================
# MODEL DIAGNOSTICS / REGIME / TEXT
# ============================================================
def oos_diagnostics(wk: pd.DataFrame):
    x = wk.dropna(subset=["d_hy", "pred_d_hy", "resid"])
    if x.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    err = x["resid"]
    return {
        "n": len(x),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }


def regime(df: pd.DataFrame):
    x = df.iloc[-1]
    off = int(x["vix"] >= 25) + int(x["z_credit"] >= 1) + int(x["spx_3m_ret"] <= -0.05)
    on = int(x["vix"] <= 18) + int(x["z_credit"] <= -0.5) + int(x["spx_3m_ret"] >= 0.05)
    if off >= 2:
        return "Risk-Off"
    if on >= 2:
        return "Risk-On"
    return "Neutral"


def signal_text(z: float, threshold: float):
    if pd.isna(z):
        return "Insufficient history", "The model does not yet have enough observations."
    if z >= threshold:
        return "Credit materially wide", "HY credit weakened materially more than the cross-asset backdrop would normally imply."
    if z >= 1.0:
        return "Credit moderately wide", "Credit is showing more stress than equities, volatility and rates alone would normally imply."
    if z <= -threshold:
        return "Credit materially tight", "HY credit is materially tighter than the cross-asset backdrop would normally imply."
    if z <= -1.0:
        return "Credit moderately tight", "Credit is showing less stress than the cross-asset backdrop would normally imply."
    return "Credit broadly aligned", "Credit is trading close to its normal cross-asset relationship."


def largest_driver(latest_wk: pd.Series):
    drivers = {
        "S&P 500": latest_wk.get("contrib_spx", np.nan),
        "VIX": latest_wk.get("contrib_vix", np.nan),
        "US 10Y": latest_wk.get("contrib_ust10", np.nan),
    }
    drivers = {k: v for k, v in drivers.items() if not pd.isna(v)}
    if not drivers:
        return "n/a", np.nan
    name = max(drivers, key=lambda k: abs(drivers[k]))
    return name, float(drivers[name])


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("RV Controls")
display_years = st.sidebar.slider("Chart lookback (years)", 2, 10, 7)
z_days = st.sidebar.slider("Daily stress z-score window", 126, 504, 252, step=21)
model_window = st.sidebar.slider("Regression window (weeks)", 52, 156, 104, step=13)
resid_window = st.sidebar.slider("Residual z-score window (weeks)", 26, 104, 52, step=13)
threshold = st.sidebar.slider("Backtest signal threshold (σ)", 1.0, 2.5, 1.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario shock")
scenario_spx = st.sidebar.slider("S&P weekly return", -10.0, 10.0, -2.0, 0.5) / 100
scenario_vix = st.sidebar.slider("VIX weekly change", -15.0, 20.0, 3.0, 0.5)
scenario_10y_bp = st.sidebar.slider("US 10Y weekly change (bp)", -75, 75, 10, 5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Backtest uses up to 15 years of history. Each weekly prediction is walk-forward and uses prior observations only."
)

# ============================================================
# BUILD
# ============================================================
try:
    raw_daily, source_last, optional_warnings = build_daily(ANALYSIS_HISTORY_YEARS)
    daily = add_daily_signals(raw_daily, z_days)
    weekly = build_weekly_model(daily, model_window, resid_window)
    events = backtest_events(weekly, threshold, cooldown=4)
except Exception as exc:
    st.error(f"Data/model error: {type(exc).__name__}: {exc}")
    st.stop()

# Display subsets, while model/backtest retain long history.
display_start = daily.index.max() - pd.DateOffset(years=display_years)
daily_view = daily[daily.index >= display_start].copy()
weekly_view = weekly[weekly.index >= display_start].copy()

last = daily.iloc[-1]
valid_model = weekly.dropna(subset=["resid_z"])
latest_wk = valid_model.iloc[-1] if not valid_model.empty else weekly.iloc[-1]
current_regime = regime(daily)
signal_title, signal_detail = signal_text(latest_wk.get("resid_z", np.nan), threshold)
signal_pct = percentile_rank(weekly["resid_z"], latest_wk.get("resid_z", np.nan))
model_diag = oos_diagnostics(weekly)
primary_driver, primary_driver_bp = largest_driver(latest_wk)

# Macro choices only if data actually loaded.
macro_options = {
    "VIX": ("vix", "Index"),
    "US 10Y Treasury": ("ust10", "%"),
    "Fed Funds Rate": ("fed_funds", "%"),
}
if daily["oil"].notna().any():
    macro_options["WTI Crude Oil"] = ("oil", "$ / bbl")
if daily["dxy"].notna().any():
    macro_options["US Dollar Index"] = ("dxy", "Index")

macro_choice = st.sidebar.selectbox("Macro overlay", list(macro_options.keys()))

# Current scenario estimate using latest model betas.
if latest_wk[["alpha", "beta_spx", "beta_vix", "beta_ust10"]].notna().all():
    scenario_move = (
        latest_wk["alpha"]
        + latest_wk["beta_spx"] * scenario_spx
        + latest_wk["beta_vix"] * scenario_vix
        + latest_wk["beta_ust10"] * (scenario_10y_bp / 100.0)
    )
else:
    scenario_move = np.nan

# ============================================================
# HEADER
# ============================================================
st.title("US High Yield Cross-Asset Relative Value Monitor")
st.markdown(
    '''
    <div class="small-note" style="line-height:1.7; max-width:1050px;">
    This dashboard compares movements in US high-yield credit with equities, volatility and interest rates to identify periods where credit is pricing a different level of risk from the wider market.
    It uses a rolling model to estimate the HY spread move implied by S&P 500 returns, VIX and US Treasury yields, then isolates the unexplained move as a relative-value signal.
    The aim is to highlight cross-asset dislocations worth investigating further rather than generate an automatic trading recommendation.
    </div>
    ''',
    unsafe_allow_html=True,
)
st.caption(
    f"Latest aligned market date: {daily.index.max().strftime('%d %b %Y')}  |  "
    f"Backtest history: {weekly.index.min().strftime('%Y')}–{weekly.index.max().strftime('%Y')}"
)

# ============================================================
# KPI STRIP
# ============================================================
k = st.columns(6)
k[0].metric("US HY OAS", f"{last['hy_oas']:.0f} bp")
k[1].metric("S&P 500", f"{last['spx']:,.0f}")
k[2].metric("VIX", f"{last['vix']:.1f}")
k[3].metric("US 10Y", f"{last['ust10']:.2f}%")
k[4].metric("RV Residual", fmt_num(latest_wk.get("resid_z", np.nan), "+.2f", "σ"))
k[5].metric("Regime", current_regime)

actual_weekly = latest_wk.get("d_hy", np.nan)
pred_weekly = latest_wk.get("pred_d_hy", np.nan)
resid_weekly = latest_wk.get("resid", np.nan)

st.markdown(
    f"""
    <div class="rv-banner">
        <div class="rv-kicker">Current relative-value signal</div>
        <div class="rv-head">{signal_title}</div>
        <div class="rv-copy">{signal_detail}</div>
        <div class="rv-meta">
            Residual: {fmt_num(latest_wk.get('resid_z', np.nan), '+.2f', 'σ')}
            &nbsp; | &nbsp; Historical percentile: {fmt_num(signal_pct, '.0f', 'th')}
            &nbsp; | &nbsp; Actual HY move: {fmt_num(actual_weekly, '+.1f', ' bp')}
            &nbsp; | &nbsp; Model implied: {fmt_num(pred_weekly, '+.1f', ' bp')}
            &nbsp; | &nbsp; Unexplained: {fmt_num(resid_weekly, '+.1f', ' bp')}
            &nbsp; | &nbsp; Primary model driver: {primary_driver} ({fmt_num(primary_driver_bp, '+.1f', ' bp')})
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if optional_warnings:
    st.caption("Optional data unavailable: " + "; ".join(optional_warnings))

# ============================================================
# TABS
# ============================================================
overview_tab, model_tab, backtest_tab, macro_tab, method_tab = st.tabs(
    ["Overview", "Model & Drivers", "Backtest", "Macro", "Methodology"]
)

# ============================================================
# OVERVIEW TAB
# ============================================================
with overview_tab:
    main = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.33, 0.30, 0.37],
        subplot_titles=("US High Yield OAS", "S&P 500", "Normalised Market Stress"),
    )

    main.add_trace(
        go.Scatter(x=daily_view.index, y=daily_view["hy_oas"], name="HY OAS", line=dict(color=BLUE, width=2)),
        1, 1,
    )
    main.add_trace(
        go.Scatter(x=daily_view.index, y=daily_view["spx"], name="S&P 500", line=dict(color=ORANGE, width=2)),
        2, 1,
    )
    main.add_trace(
        go.Scatter(x=daily_view.index, y=daily_view["z_credit"], name="Credit stress", line=dict(color=GREEN, width=2)),
        3, 1,
    )
    main.add_trace(
        go.Scatter(x=daily_view.index, y=daily_view["z_equity"], name="Equity stress", line=dict(color=PURPLE, width=2)),
        3, 1,
    )
    main.add_hline(y=0, row=3, col=1, line_width=1, line_color=MUTED)

    # Major stress periods: context markers, not model inputs.
    event_markers = [
        (pd.Timestamp("2020-03-20"), "COVID shock"),
        (pd.Timestamp("2022-06-15"), "2022 rates repricing"),
        (pd.Timestamp("2023-03-10"), "US banking stress"),
    ]
    for date, label in event_markers:
        if daily_view.index.min() <= date <= daily_view.index.max():
            main.add_vline(x=date, line_width=1, line_dash="dot", line_color="#536473")
            main.add_annotation(
                x=date, y=1.02, yref="paper", text=label, showarrow=False,
                textangle=-90, font=dict(size=10, color=MUTED),
            )

    main.update_yaxes(title_text="bp", row=1, col=1)
    main.update_yaxes(title_text="Index", row=2, col=1)
    main.update_yaxes(title_text="σ", row=3, col=1)
    style_figure(main, height=760)
    st.plotly_chart(main, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HY spread percentile", fmt_num(percentile_rank(daily["hy_oas"], last["hy_oas"]), ".0f", "th"))
    c2.metric("Credit stress", fmt_num(last_value(daily["z_credit"]), "+.2f", "σ"))
    c3.metric("Equity stress", fmt_num(last_value(daily["z_equity"]), "+.2f", "σ"))
    c4.metric("Credit − equity stress", fmt_num(last["stress_gap"], "+.2f", "σ"))

# ============================================================
# MODEL & DRIVERS TAB
# ============================================================
with model_tab:
    st.subheader("Cross-Asset Credit Residual Model")
    left, right = st.columns([1.45, 1])

    model_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Actual vs Model-Predicted Weekly HY Spread Change",
            "Standardised Unexplained Credit Move",
        ),
    )
    model_fig.add_trace(
        go.Bar(x=weekly_view.index, y=weekly_view["d_hy"], name="Actual ΔHY OAS", marker_color=BLUE, opacity=0.55),
        1, 1,
    )
    model_fig.add_trace(
        go.Scatter(x=weekly_view.index, y=weekly_view["pred_d_hy"], name="Model predicted", line=dict(color=ORANGE, width=2)),
        1, 1,
    )
    model_fig.add_trace(
        go.Scatter(x=weekly_view.index, y=weekly_view["resid_z"], name="Residual z", line=dict(color=GREEN, width=2)),
        2, 1,
    )
    model_fig.add_hline(y=threshold, row=2, col=1, line_dash="dash", line_width=1, line_color=YELLOW)
    model_fig.add_hline(y=-threshold, row=2, col=1, line_dash="dash", line_width=1, line_color=YELLOW)
    model_fig.add_hline(y=0, row=2, col=1, line_width=1, line_color=MUTED)

    if not events.empty:
        shown_events = events[events["date"] >= weekly_view.index.min()]
        for label, symbol, color in [
            ("Credit wide", "triangle-down", RED),
            ("Credit tight", "triangle-up", GREEN),
        ]:
            e = shown_events[shown_events["signal"] == label]
            model_fig.add_trace(
                go.Scatter(
                    x=e["date"], y=e["resid_z"], mode="markers", name=label,
                    marker=dict(size=9, symbol=symbol, color=color),
                ),
                2, 1,
            )

    model_fig.update_yaxes(title_text="bp/week", row=1, col=1)
    model_fig.update_yaxes(title_text="σ", row=2, col=1)
    style_figure(model_fig, height=650)
    left.plotly_chart(model_fig, width="stretch")

    latest_train_r2 = last_value(weekly["train_r2"])
    diag = pd.DataFrame({
        "Metric": [
            "Training-window R²",
            "Walk-forward predictions",
            "OOS MAE",
            "OOS RMSE",
            "OOS bias",
            "β: S&P weekly return",
            "β: weekly VIX change",
            "β: weekly 10Y change",
        ],
        "Value": [
            fmt_num(latest_train_r2, ".2f"),
            str(model_diag["n"]),
            fmt_num(model_diag["mae"], ".1f", " bp"),
            fmt_num(model_diag["rmse"], ".1f", " bp"),
            fmt_num(model_diag["bias"], "+.1f", " bp"),
            fmt_num(last_value(weekly["beta_spx"]), ".1f"),
            fmt_num(last_value(weekly["beta_vix"]), ".2f"),
            fmt_num(last_value(weekly["beta_ust10"]), ".1f"),
        ],
    })
    right.markdown("#### Model diagnostics")
    right.dataframe(diag, hide_index=True, width="stretch")
    right.markdown(
        """
        <div class="driver-box">
        <b>Why this is defensible</b><br>
        Each weekly prediction is made with a regression fitted only on earlier observations.
        MAE and RMSE therefore measure genuine walk-forward prediction error rather than an in-sample fit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What drove the latest HY move?")
    wf = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["Intercept", "S&P", "VIX", "US 10Y", "Unexplained", "Actual ΔHY"],
            y=[
                latest_wk.get("contrib_intercept", 0),
                latest_wk.get("contrib_spx", 0),
                latest_wk.get("contrib_vix", 0),
                latest_wk.get("contrib_ust10", 0),
                latest_wk.get("resid", 0),
                latest_wk.get("d_hy", 0),
            ],
            connector={"line": {"color": MUTED}},
            increasing={"marker": {"color": RED}},
            decreasing={"marker": {"color": GREEN}},
            totals={"marker": {"color": BLUE}},
            textposition="outside",
            texttemplate="%{y:+.1f} bp",
        )
    )
    wf.update_layout(title="Latest Weekly HY Spread Move: Driver Decomposition", yaxis_title="Basis points")
    style_figure(wf, height=430, legend=False)
    st.plotly_chart(wf, width="stretch")

    st.subheader("Scenario Shock Tool")
    s1, s2 = st.columns([1, 1.3])
    s1.metric("Model-implied HY move", fmt_num(scenario_move, "+.1f", " bp"))
    s1.markdown(
        f"""
        Scenario entered in the sidebar:
        **S&P {scenario_spx:+.1%}**, **VIX {scenario_vix:+.1f} points**, **US 10Y {scenario_10y_bp:+d} bp**.

        This uses the **latest rolling coefficients**, so it is a local sensitivity estimate rather than a forecast.
        """
    )

    coef_view = weekly_view[["beta_spx", "beta_vix", "beta_ust10"]].dropna()
    coef_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=("β to S&P return", "β to VIX change", "β to US 10Y change"))
    coef_fig.add_trace(go.Scatter(x=coef_view.index, y=coef_view["beta_spx"], line=dict(color=ORANGE, width=2), name="S&P β"), 1, 1)
    coef_fig.add_trace(go.Scatter(x=coef_view.index, y=coef_view["beta_vix"], line=dict(color=GREEN, width=2), name="VIX β"), 2, 1)
    coef_fig.add_trace(go.Scatter(x=coef_view.index, y=coef_view["beta_ust10"], line=dict(color=PURPLE, width=2), name="10Y β"), 3, 1)
    style_figure(coef_fig, height=500, legend=False)
    s2.plotly_chart(coef_fig, width="stretch")

# ============================================================
# BACKTEST TAB
# ============================================================
with backtest_tab:
    st.subheader("Historical Threshold-Crossing Backtest")
    st.caption(
        "An event is recorded only when the residual first enters the ±threshold zone. "
        "A four-week cooldown reduces duplicate signals from the same dislocation."
    )

    stats4 = backtest_by_side(events, 4)
    stats12 = backtest_by_side(events, 12)

    def display_stats_table(stats: pd.DataFrame, horizon: int):
        out = stats.copy()
        out["Hit rate"] = out["Hit rate"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")
        for col in ["Avg OAS move", "Median OAS move"]:
            out[col] = out[col].map(lambda x: "n/a" if pd.isna(x) else f"{x:+.0f} bp")
        for col in ["Avg S&P return", "Median S&P return"]:
            out[col] = out[col].map(lambda x: "n/a" if pd.isna(x) else f"{x:+.1%}")
        st.markdown(f"#### {horizon}-week outcomes")
        st.dataframe(out, hide_index=True, width="stretch")

    bleft, bright = st.columns(2)
    with bleft:
        display_stats_table(stats4, 4)
    with bright:
        display_stats_table(stats12, 12)

    if events.empty:
        st.warning(
            "No threshold-crossing events were found even after the event-detection fix. "
            "Lower the threshold or shorten the residual z-score window."
        )
    else:
        st.markdown(f"**Total unique signal events:** {len(events)}")

        # Forward outcome distributions by signal side.
        dist = make_subplots(rows=1, cols=2, subplot_titles=("4-week HY OAS move", "12-week HY OAS move"))
        for side, color in [("Credit wide", RED), ("Credit tight", GREEN)]:
            e = events[events["signal"] == side]
            dist.add_trace(go.Box(y=e["oas_4w"], name=side, marker_color=color, boxmean=True), 1, 1)
            dist.add_trace(go.Box(y=e["oas_12w"], name=side, marker_color=color, boxmean=True, showlegend=False), 1, 2)
        dist.update_yaxes(title_text="bp", row=1, col=1)
        dist.update_yaxes(title_text="bp", row=1, col=2)
        style_figure(dist, height=430)
        st.plotly_chart(dist, width="stretch")

        event_table = events.copy()
        event_table["date"] = pd.to_datetime(event_table["date"]).dt.strftime("%Y-%m-%d")
        event_table["resid_z"] = event_table["resid_z"].map(lambda x: f"{x:+.2f}σ")
        for col in ["oas_4w", "oas_12w"]:
            event_table[col] = event_table[col].map(lambda x: "" if pd.isna(x) else f"{x:+.0f} bp")
        for col in ["spx_4w", "spx_12w"]:
            event_table[col] = event_table[col].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
        for col in ["mr_4w", "mr_12w"]:
            event_table[col] = event_table[col].map(lambda x: "" if pd.isna(x) else ("Yes" if x else "No"))

        event_table = event_table.rename(columns={
            "date": "Date",
            "signal": "Signal",
            "resid_z": "Entry z",
            "oas_4w": "HY OAS +4w",
            "spx_4w": "S&P +4w",
            "mr_4w": "MR +4w",
            "oas_12w": "HY OAS +12w",
            "spx_12w": "S&P +12w",
            "mr_12w": "MR +12w",
        })
        st.dataframe(
            event_table[["Date", "Signal", "Entry z", "HY OAS +4w", "S&P +4w", "MR +4w", "HY OAS +12w", "S&P +12w", "MR +12w"]].tail(30),
            hide_index=True,
            width="stretch",
        )

    st.info(
        "Interpret the backtest as descriptive evidence, not a trading rule. A small number of events, regime shifts, "
        "transaction costs and implementation choices can materially change results."
    )

# ============================================================
# MACRO TAB
# ============================================================
with macro_tab:
    st.subheader("Macro Driver Overlay")
    macro_col, macro_unit = macro_options[macro_choice]

    macro = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("US HY OAS", macro_choice),
    )
    macro.add_trace(go.Scatter(x=daily_view.index, y=daily_view["hy_oas"], name="HY OAS", line=dict(color=BLUE, width=2)), 1, 1)
    macro.add_trace(go.Scatter(x=daily_view.index, y=daily_view[macro_col], name=macro_choice, line=dict(color=ORANGE, width=2)), 2, 1)
    macro.update_yaxes(title_text="bp", row=1, col=1)
    macro.update_yaxes(title_text=macro_unit, row=2, col=1)
    style_figure(macro, height=560, legend=False)
    st.plotly_chart(macro, width="stretch")

    corr_source = weekly[["d_hy", "spx_ret", "d_vix", "d_ust10"]].dropna()
    if len(corr_source) >= 20:
        corr = corr_source.corr()
        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=["ΔHY OAS", "S&P return", "ΔVIX", "Δ10Y"],
                y=["ΔHY OAS", "S&P return", "ΔVIX", "Δ10Y"],
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r",
                text=np.round(corr.values, 2),
                texttemplate="%{text:.2f}",
                colorbar=dict(title="Correlation"),
            )
        )
        heat.update_layout(title="Weekly Cross-Asset Correlation Matrix")
        style_figure(heat, height=460, legend=False)
        st.plotly_chart(heat, width="stretch")

# ============================================================
# METHODOLOGY TAB
# ============================================================
with method_tab:
    st.subheader("Model specification")
    st.code("ΔHY OAS = α + β1(S&P weekly return) + β2(ΔVIX) + β3(ΔUS10Y) + ε", language="text")

    st.subheader("Data sources")
    source_df = pd.DataFrame({
        "Series": list(source_last.keys()),
        "Latest raw observation": [
            "n/a" if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")
            for v in source_last.values()
        ],
    })
    st.dataframe(source_df, hide_index=True, width="stretch")
    st.caption(
        "US HY OAS, US 10Y and Fed Funds are loaded from FRED. S&P 500, VIX and optional market overlays are loaded from Yahoo Finance."
    )

    st.subheader("Limitations")
    st.markdown(
        """
        The model is useful for spotting unusual moves, but there are a few things I would keep in mind. HY OAS is not a pure measure of default risk because it also reflects things like liquidity and risk premia. The relationship between credit, equities, volatility and rates can also change depending on the market environment, so a signal that worked historically may not behave the same way in the future. The backtest has a limited number of independent stress periods and it does not include transaction costs, financing or execution. I would therefore use the dashboard as a way to identify where something looks unusual and investigate it further, rather than treat the signal as an automatic trade.
        """
    )

    with st.expander("Download model output"):
        output = weekly[[
            "hy_oas", "spx", "vix", "ust10", "d_hy", "pred_d_hy", "resid", "resid_z",
            "contrib_spx", "contrib_vix", "contrib_ust10", "train_r2"
        ]].copy()
        st.dataframe(output.tail(60), width="stretch")
        st.download_button(
            "Download CSV",
            output.to_csv().encode("utf-8"),
            "us_hy_cross_asset_rv_model.csv",
            "text/csv",
        )

st.caption("Research dashboard only. Not investment advice.")
