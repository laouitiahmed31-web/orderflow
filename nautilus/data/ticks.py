"""
Convert recorded tick Parquet (legacy Redis recorder format) into Nautilus ``TradeTick``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler


def trade_tick_to_side_dict(tick: TradeTick) -> dict[str, Any] | None:
    """Map Nautilus trade tick to orderflow engine tick dict."""
    ts_ms = int(tick.ts_event / 1_000_000)
    price = float(tick.price)
    qty = float(tick.size)
    
    if tick.aggressor_side == AggressorSide.BUYER:
        side = "BUY"
    elif tick.aggressor_side == AggressorSide.SELLER:
        side = "SELL"
    else:
        # Fallback: infer from previous price or use default
        # For now, default to BUY if aggressor side not set (better than rejecting)
        side = "BUY"
    
    return {"ts": ts_ms, "price": price, "qty": qty, "side": side}


def parquet_ticks_to_trade_ticks(
    parquet_path: str | Path,
    instrument: Instrument,
    *,
    ts_init_delta: int = 0,
    start_ts_ms: int | None = None,
    end_ts_ms: int | None = None,
) -> list:
    """
    Load ``tick_recorder.py`` Parquet (columns: ts, price, qty, side, agg_id).

    Returns
    -------
    list[TradeTick]
    """
    path = Path(parquet_path)
    # Column projection keeps Parquet reads fast for large files.
    df = pd.read_parquet(path, columns=["ts", "price", "qty", "side", "agg_id"])
    if df.empty:
        return []

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["ts", "price", "qty"]).copy()
    if df.empty:
        return []
    # Normalize ts to milliseconds for downstream filtering and conversion.
    ts_sample = int(df["ts"].dropna().iloc[0])
    if ts_sample >= 10**18:      # ns
        df["ts"] = df["ts"] // 1_000_000
    elif ts_sample >= 10**15:    # us
        df["ts"] = df["ts"] // 1_000
    elif ts_sample >= 10**12:    # ms
        pass
    else:                        # seconds
        df["ts"] = df["ts"] * 1_000
    if start_ts_ms is not None:
        df = df[df["ts"] >= start_ts_ms]
    if end_ts_ms is not None:
        df = df[df["ts"] <= end_ts_ms]
    if df.empty:
        return []

    # Stable sort preserves file order for equal timestamps.
    df = df.sort_values(["ts", "agg_id"], kind="mergesort")

    # Vectorized side mapping (no row-wise apply).
    side_raw = df["side"].astype(str).str.lower()
    is_buy = side_raw.isin(["buy", "b", "true", "1"])
    is_sell = side_raw.isin(["sell", "s", "false", "0"])
    df["side"] = np.where(is_sell, "SELL", np.where(is_buy, "BUY", "BUY"))

    df["trade_id"] = df["agg_id"].astype(str)
    df["quantity"] = (df["qty"] * 1_000_000).round().clip(lower=1).astype("int64")

    # Ensure strictly increasing timestamps for wrangler/index.
    tie_idx = df.groupby("ts", sort=False).cumcount()
    ts_ns = df["ts"].astype("int64") * 1_000_000 + tie_idx
    df["timestamp"] = pd.to_datetime(ts_ns, unit="ns", utc=True)
    df = df.drop_duplicates(subset=["timestamp", "trade_id"], keep="first").copy()
    df = df.sort_values("timestamp", kind="mergesort")
    print(f"  Loaded {len(df):,} unique ticks")

    df = df.set_index("timestamp")

    wrangler = TradeTickDataWrangler(instrument=instrument)
    return wrangler.process(df, ts_init_delta=ts_init_delta)
