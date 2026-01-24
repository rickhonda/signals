#!/usr/bin/env python3
"""
signals.py — minimal, practical "signal builder + baseline/scoring" CLI

Works on a generic event-table CSV like events.csv.
Outputs long-form signals, and can run EWMA + rolling MAD scoring (+ optional CUSUM).

Install deps:
  pip install pandas pyyaml pyarrow python-dateutil

Examples:
  python signals.py build  --input events.csv --channels channels.yaml --hop 1m --window 5m --out signals.parquet
  python signals.py detect --input signals.parquet --alpha 0.05 --score-window 240 --threshold 6 --cusum --out anomalies.parquet
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


# ----------------------------
# Utilities
# ----------------------------

def parse_duration_to_timedelta(s: str) -> pd.Timedelta:
    """Parse strings like '30s', '5m', '2h', '1d' into pandas Timedelta."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([smhd])", s)
    if not m:
        raise ValueError(f"Invalid duration '{s}'. Use like 30s/5m/2h/1d.")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "s":
        return pd.Timedelta(seconds=n)
    if unit == "m":
        return pd.Timedelta(minutes=n)
    if unit == "h":
        return pd.Timedelta(hours=n)
    if unit == "d":
        return pd.Timedelta(days=n)
    raise ValueError(f"Unsupported duration unit in '{s}'.")

def safe_series_key(group_by: List[str], row: pd.Series) -> str:
    if not group_by:
        return "global"
    parts = []
    for f in group_by:
        v = row.get(f, None)
        if pd.isna(v):
            v = "null"
        parts.append(f"{f}={v}")
    return "|".join(parts)


# ----------------------------
# Channel spec
# ----------------------------

@dataclass(frozen=True)
class ChannelSpec:
    name: str
    filter_expr: str
    group_by: List[str]
    measure: str = "count"  # v1: only count
    top_k: Optional[int] = None
    min_events: Optional[int] = None


def load_channels_yaml(path: str) -> Tuple[str, List[ChannelSpec]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    time_field = data.get("time_field", "timestamp")
    chs = []
    for ch in data.get("channels", []):
        chs.append(
            ChannelSpec(
                name=ch["name"],
                filter_expr=ch.get("filter", "True"),
                group_by=list(ch.get("group_by", [])),
                measure=ch.get("measure", "count"),
                top_k=ch.get("top_k", None),
                min_events=ch.get("min_events", None),
            )
        )
    if not chs:
        raise ValueError(f"No channels found in {path}.")
    return time_field, chs


# ----------------------------
# Signal building
# ----------------------------

def build_signals_longform(
    events: pd.DataFrame,
    time_col: str,
    channels: List[ChannelSpec],
    hop: pd.Timedelta,
    window: pd.Timedelta,
) -> pd.DataFrame:
    """
    Build long-form signals:
      columns: ts, channel, series_key, value

    Semantics:
    - hop defines evaluation grid
    - window defines lookback window
      - if window == hop: fixed bins
      - if window > hop: sliding window evaluated at hop times
    """
    if time_col not in events.columns:
        raise ValueError(f"Input missing time column '{time_col}'.")

    df = events.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    out_rows: List[Dict[str, Any]] = []

    # Determine global grid for consistent alignment
    tmin = df[time_col].min()
    tmax = df[time_col].max()
    if pd.isna(tmin) or pd.isna(tmax):
        raise ValueError("No valid timestamps found after parsing.")

    # Align grid start to hop boundary (simple floor)
    start = (tmin.floor(freq=hop) if hasattr(tmin, "floor") else tmin)
    end = (tmax.ceil(freq=hop) if hasattr(tmax, "ceil") else tmax)

    grid = pd.date_range(start=start, end=end, freq=hop, tz="UTC")
    # We'll evaluate at grid points as right-edges: [t-window, t)
    # For window==hop and count, that's equivalent to bin count.

    for spec in channels:
        if spec.measure != "count":
            raise ValueError(f"v1 supports only measure=count. Got {spec.measure} in {spec.name}")

        # Filter
        try:
            sdf = df.query(spec.filter_expr, engine="python")
        except Exception as e:
            raise ValueError(f"Bad filter in channel '{spec.name}': {spec.filter_expr}\n{e}")

        if sdf.empty:
            continue

        # If no group-by: compute global series
        if not spec.group_by:
            # count events in each sliding window evaluated at grid times
            # Use searchsorted on timestamps for speed and clear semantics.
            ts = sdf[time_col].to_numpy()
            # ensure numpy datetime64[ns]
            ts = ts.astype("datetime64[ns]")

            for t in grid:
                t_ns = t.to_datetime64()
                left = (t - window).to_datetime64()
                # count in [left, t)
                lo = ts.searchsorted(left, side="left")
                hi = ts.searchsorted(t_ns, side="left")
                out_rows.append(
                    {"ts": t, "channel": spec.name, "series_key": "global", "value": float(hi - lo)}
                )
            continue

        # Grouped: compute counts per group in each window
        # Strategy:
        # 1) create a series_key per event row
        # 2) optionally restrict to top_k groups by total count
        tmp = sdf.copy()
        for col in spec.group_by:
            if col not in tmp.columns:
                tmp[col] = None  # allow missing fields to become "null"
        tmp["series_key"] = tmp.apply(lambda r: safe_series_key(spec.group_by, r), axis=1)

        # Top-K / min_events filtering by total count
        group_counts = tmp["series_key"].value_counts()
        if spec.min_events is not None:
            group_counts = group_counts[group_counts >= int(spec.min_events)]
        if spec.top_k is not None:
            group_counts = group_counts.head(int(spec.top_k))

        keep = set(group_counts.index.tolist())
        tmp = tmp[tmp["series_key"].isin(keep)]
        if tmp.empty:
            continue

        # Pre-split timestamps per series_key for fast window counting
        ts_by_key: Dict[str, Any] = {}
        for key, g in tmp.groupby("series_key"):
            arr = g[time_col].to_numpy().astype("datetime64[ns]")
            arr.sort()
            ts_by_key[key] = arr

        for key, arr in ts_by_key.items():
            for t in grid:
                t_ns = t.to_datetime64()
                left = (t - window).to_datetime64()
                lo = arr.searchsorted(left, side="left")
                hi = arr.searchsorted(t_ns, side="left")
                out_rows.append(
                    {"ts": t, "channel": spec.name, "series_key": key, "value": float(hi - lo)}
                )

    if not out_rows:
        return pd.DataFrame(columns=["ts", "channel", "series_key", "value"])

    out = pd.DataFrame(out_rows)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values(["channel", "series_key", "ts"]).reset_index(drop=True)
    return out


# ----------------------------
# Detection: EWMA baseline + rolling MAD score + optional CUSUM
# ----------------------------

def ewma_baseline(x: pd.Series, alpha: float) -> pd.Series:
    # pandas ewm uses span/alpha; here we use alpha directly
    return x.ewm(alpha=alpha, adjust=False).mean()


def rolling_mad_zscore(r: pd.Series, window: int, eps: float = 1e-9, mad_floor: float = 1.0) -> pd.Series:
    """
    Robust z-score using rolling median + rolling MAD.
    z(t) = (r(t) - median) / max(1.4826*MAD, mad_floor)
    """
    med = r.rolling(window, min_periods=window).median()
    mad = (r - med).abs().rolling(window, min_periods=window).median()
    scale = 1.4826 * mad
    scale = scale.clip(lower=mad_floor)  # avoid blowing up
    z = (r - med) / (scale + eps)
    return z


def cusum_pos(z: pd.Series, k: float = 0.0) -> pd.Series:
    """
    One-sided positive CUSUM on z:
      S_t = max(0, S_{t-1} + (z_t - k))
    """
    s = []
    acc = 0.0
    for v in z.fillna(0.0).to_list():
        acc = max(0.0, acc + (float(v) - k))
        s.append(acc)
    return pd.Series(s, index=z.index, dtype="float64")


def detect_on_signals(
    signals: pd.DataFrame,
    alpha: float,
    score_window: int,
    threshold: float,
    use_cusum: bool,
    cusum_k: float,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    required = {"ts", "channel", "series_key", "value"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Signals input missing columns: {sorted(missing)}")

    df = signals.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["channel", "series_key", "ts"])

    out_parts = []
    for (ch, key), g in df.groupby(["channel", "series_key"], sort=False):
        x = g["value"].astype(float)
        base = ewma_baseline(x, alpha=alpha)
        resid = x - base
        z = rolling_mad_zscore(resid, window=score_window)

        part = g.copy()
        part["baseline"] = base
        part["residual"] = resid
        part["score"] = z

        if use_cusum:
            part["cusum"] = cusum_pos(part["score"], k=cusum_k)
            part["is_alert"] = (part["score"].abs() >= threshold) | (part["cusum"] >= threshold)
        else:
            part["cusum"] = float("nan")
            part["is_alert"] = (part["score"].abs() >= threshold)

        out_parts.append(part)

    out = pd.concat(out_parts, ignore_index=True)
    return out


# ----------------------------
# CLI
# ----------------------------

def cmd_build(args: argparse.Namespace) -> int:
    hop = parse_duration_to_timedelta(args.hop)
    window = parse_duration_to_timedelta(args.window)

    time_field, channels = load_channels_yaml(args.channels)

    # Read events
    events = pd.read_csv(args.input)
    # Ensure expected columns exist (but keep generic)
    if time_field not in events.columns:
        raise SystemExit(f"Time field '{time_field}' not found in {args.input} columns: {list(events.columns)}")

    signals = build_signals_longform(
        events=events,
        time_col=time_field,
        channels=channels,
        hop=hop,
        window=window,
    )

    if args.out.endswith(".csv"):
        signals.to_csv(args.out, index=False)
    else:
        signals.to_parquet(args.out, index=False)
    print(f"Wrote {len(signals):,} signal rows to {args.out}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    if args.input.endswith(".csv"):
        signals = pd.read_csv(args.input)
    else:
        signals = pd.read_parquet(args.input)

    out = detect_on_signals(
        signals=signals,
        alpha=float(args.alpha),
        score_window=int(args.score_window),
        threshold=float(args.threshold),
        use_cusum=bool(args.cusum),
        cusum_k=float(args.cusum_k),
    )

    if args.out.endswith(".csv"):
        out.to_csv(args.out, index=False)
    else:
        out.to_parquet(args.out, index=False)

    # Also write a small "alerts-only" file if requested
    if args.alerts_out:
        alerts = out[out["is_alert"] == True].copy()
        if args.alerts_out.endswith(".csv"):
            alerts.to_csv(args.alerts_out, index=False)
        else:
            alerts.to_parquet(args.alerts_out, index=False)
        print(f"Wrote {len(alerts):,} alert rows to {args.alerts_out}")

    print(f"Wrote {len(out):,} analyzed rows to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signals", description="Generic signal builder + baseline/scoring CLI (v1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build long-form signals from an event-table CSV")
    pb.add_argument("--input", required=True, help="Input event table (CSV) e.g. events.csv")
    pb.add_argument("--channels", required=True, help="YAML channel pack")
    pb.add_argument("--hop", required=True, help="Evaluation step, e.g. 1m")
    pb.add_argument("--window", required=True, help="Lookback window, e.g. 5m")
    pb.add_argument("--out", required=True, help="Output signals (.parquet recommended)")
    pb.set_defaults(func=cmd_build)

    pdx = sub.add_parser("detect", help="Compute baseline/residual/score (+ optional CUSUM) over long-form signals")
    pdx.add_argument("--input", required=True, help="Input signals (.parquet or .csv)")
    pdx.add_argument("--alpha", default=0.05, help="EWMA alpha (default 0.05)")
    pdx.add_argument("--score-window", default=240, help="Rolling MAD window in points (default 240)")
    pdx.add_argument("--threshold", default=6.0, help="Score/CUSUM threshold (default 6)")
    pdx.add_argument("--cusum", action="store_true", help="Enable one-sided positive CUSUM on score")
    pdx.add_argument("--cusum-k", default=0.0, help="CUSUM slack k (default 0.0)")
    pdx.add_argument("--out", required=True, help="Output analyzed signals (.parquet recommended)")
    pdx.add_argument("--alerts-out", default=None, help="Optional separate output of alert rows only")
    pdx.set_defaults(func=cmd_detect)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

