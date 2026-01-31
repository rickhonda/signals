import streamlit as st
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# --- config ---
st.set_page_config(layout="wide")
st.title("Signal Tuning Workbench")

# --- repo root resolution ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_PATH = REPO_ROOT / "examples" / "k-test" / "signals.parquet"

# --- load signals ---
@st.cache_data
def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df

if not SIGNALS_PATH.exists():
    st.error(f"Signals file not found:\n{SIGNALS_PATH}")
    st.stop()

signals = load_signals(SIGNALS_PATH)

# --- sidebar controls ---
st.sidebar.header("Selection")

channel = st.sidebar.selectbox(
    "Channel",
    sorted(signals["channel"].unique())
)

series_keys = (
    signals.loc[signals["channel"] == channel, "series_key"]
    .astype(str)
    .unique()
)

series_key = st.sidebar.selectbox(
    "Series key",
    sorted(series_keys)
)

st.sidebar.header("Tuning")

alpha = st.sidebar.slider(
    "EWMA alpha",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f",
)

score_window = st.sidebar.slider(
    "MAD window",
    min_value=10,
    max_value=1000,
    value=240,
    step=10,
)

threshold = st.sidebar.slider(
    "Alert threshold",
    min_value=1.0,
    max_value=15.0,
    value=6.0,
    step=0.5,
)

# --- extract series ---
mask = (
    (signals["channel"] == channel) &
    (signals["series_key"].astype(str) == series_key)
)

s = (
    signals.loc[mask]
    .sort_values("ts")
    .set_index("ts")
)

x = s["value"].astype(float)

# --- compute baseline + score using your seams ---
from signals import (
    ewma_baseline,
    rolling_mad_zscore,
)

baseline = ewma_baseline(x, alpha=alpha)
residual = x - baseline
score = rolling_mad_zscore(
    residual,
    window=int(score_window),
    mad_floor=1.0,
)

# --- plots ---
st.subheader(f"{channel} | {series_key}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Signal & Baseline")
    st.line_chart(
        pd.DataFrame(
            {"x(t)": x, "baseline": baseline}
        )
    )

with col2:
    st.markdown("### Score")
    st.line_chart(score)

st.markdown("### Score with Threshold")
st.line_chart(
    pd.DataFrame(
        {
            "score": score,
            "threshold": threshold,
        }
    )
)

# --- quick stats ---
st.markdown("### Summary")
st.write(
    {
        "points": len(score),
        "max_score": float(score.max()),
        "alerts": int((score.abs() >= threshold).sum()),
    }
)

