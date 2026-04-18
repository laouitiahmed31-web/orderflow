#!/usr/bin/env python3
"""
backtest.py — Live Strategy Backtester
============================================
Uses the ACTUAL live strategy (OrderflowStrategy) for backtesting.

When you change parameters in live.yaml, backtest behavior changes identically.

DATA PIPELINE:
  raw tick CSV → candles → OrderflowStrategy → trades

Usage:
  python backtest.py --config nautilus/config/profiles/live.yaml --ticks data/btcusdt_ticks.csv
  python backtest.py --demo --config nautilus/config/profiles/live.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pyarrow.dataset as ds

# ── Import the actual live strategy and its components ────────────────────────────────
try:
    from nautilus.config.loader import load_orderflow_config
    from nautilus.config.schema import orderflow_strategy_config_from_stack
    from nautilus.strategy.orderflow_strategy import OrderflowStrategy
    _LIVE_STRATEGY_AVAILABLE = True
except ImportError as e:
    _LIVE_STRATEGY_AVAILABLE = False
    print(f"⚠  Live strategy not available: {e}")
    print("   Ensure nautilus/ directory is available.")
    sys.exit(1)

# Define _INDICATORS_AVAILABLE for compatibility with existing functions
_INDICATORS_AVAILABLE = True
BacktestConfig = Any
CandleFlow = Any

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):  # type: ignore[misc]
        def _inner(fn):
            return fn
        return _inner


# ══════════════════════════════════════════════════════════════════════════════
# RAW TICK
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Tick:
    ts:    int
    price: float
    qty:   float
    side:  str   # 'buy' or 'sell'


# ══════════════════════════════════════════════════════════════════════════════
# TRADE RECORD
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    trade_id:      int
    entry_time:    datetime
    exit_time:     Optional[datetime] = None
    entry_price:   float = 0.0
    exit_price:    float = 0.0
    qty:           float = 0.0
    capital_used:  float = 0.0
    entry_reason:  str   = ""
    exit_reason:   str   = ""
    pnl_pct:       float = 0.0
    pnl_usdt:      float = 0.0
    duration_mins: float = 0.0
    conditions:    dict  = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# TICK LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _range_ms(
    start: str = None,
    end: str = None,
    start_time: str = None,
    end_time: str = None,
) -> tuple[int | None, int | None]:
    sd_ms = None
    ed_ms = None
    if start:
        sd = datetime.strptime(start, "%Y-%m-%d")
        if start_time:
            h, m = map(int, start_time.split(":"))
            sd = sd.replace(hour=h, minute=m, second=0, microsecond=0)
        sd_ms = int(sd.timestamp() * 1000)
    if end:
        ed = datetime.strptime(end, "%Y-%m-%d")
        if end_time:
            h, m = map(int, end_time.split(":"))
            ed = ed.replace(hour=h, minute=m, second=59, microsecond=999999)
        else:
            ed = ed.replace(hour=23, minute=59, second=59, microsecond=999999)
        ed_ms = int(ed.timestamp() * 1000)
    return sd_ms, ed_ms


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("**/*.parquet") if p.is_file())


def estimate_parquet_ticks(path: str, start_ms: int | None, end_ms: int | None) -> int:
    files = _parquet_files(Path(path))
    if not files:
        return 0
    dataset = ds.dataset([str(p) for p in files], format="parquet")
    filt = None
    if start_ms is not None:
        filt = ds.field("ts") >= start_ms
    if end_ms is not None:
        end_filter = ds.field("ts") <= end_ms
        filt = end_filter if filt is None else (filt & end_filter)
    scanner = dataset.scanner(columns=["ts"], filter=filt)
    return sum(batch.num_rows for batch in scanner.to_batches())


@njit(cache=True)
def _compress_ticks_numba(
    ts: np.ndarray,
    px: np.ndarray,
    qty: np.ndarray,
    side_code: np.ndarray,
    bucket_ms: int,
):
    n = ts.shape[0]
    out_ts = np.empty(n, dtype=np.int64)
    out_px = np.empty(n, dtype=np.float64)
    out_qty = np.empty(n, dtype=np.float64)
    out_side = np.empty(n, dtype=np.int8)
    k = 0

    cur_bucket = -1
    cur_side = -1
    qty_sum = 0.0
    notional_sum = 0.0
    for i in range(n):
        b = (ts[i] // bucket_ms) * bucket_ms
        s = side_code[i]
        if cur_bucket == -1:
            cur_bucket = b
            cur_side = s
        if b != cur_bucket or s != cur_side:
            if qty_sum > 0.0:
                out_ts[k] = cur_bucket
                out_px[k] = notional_sum / qty_sum
                out_qty[k] = qty_sum
                out_side[k] = cur_side
                k += 1
            cur_bucket = b
            cur_side = s
            qty_sum = 0.0
            notional_sum = 0.0
        qty_sum += qty[i]
        notional_sum += px[i] * qty[i]

    if qty_sum > 0.0:
        out_ts[k] = cur_bucket
        out_px[k] = notional_sum / qty_sum
        out_qty[k] = qty_sum
        out_side[k] = cur_side
        k += 1
    return out_ts[:k], out_px[:k], out_qty[:k], out_side[:k]


def stream_parquet_tick_batches(
    parquet_path: str,
    *,
    start: str | None = None,
    end: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    batch_size: int = 200_000,
    compress_ms: int = 0,
) -> Iterator[list[Tick]]:
    files = _parquet_files(Path(parquet_path))
    if not files:
        return

    sd_ms, ed_ms = _range_ms(start, end, start_time, end_time)
    dataset = ds.dataset([str(p) for p in files], format="parquet")
    filt = None
    if sd_ms is not None:
        filt = ds.field("ts") >= sd_ms
    if ed_ms is not None:
        end_filter = ds.field("ts") <= ed_ms
        filt = end_filter if filt is None else (filt & end_filter)

    scanner = dataset.scanner(
        columns=["ts", "price", "qty", "side"],
        filter=filt,
        batch_size=batch_size,
    )

    for rb in scanner.to_batches():
        ts = rb.column("ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        px = rb.column("price").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        qty = rb.column("qty").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        side_arr = rb.column("side").to_pylist()
        side_code = np.fromiter(
            (1 if str(s).upper().startswith("B") else -1 for s in side_arr),
            dtype=np.int8,
            count=len(side_arr),
        )

        if compress_ms > 0 and len(ts) > 0:
            ts, px, qty, side_code = _compress_ticks_numba(ts, px, qty, side_code, compress_ms)

        batch_ticks = [
            Tick(
                ts=int(ts[i]),
                price=float(px[i]),
                qty=float(qty[i]),
                side="BUY" if side_code[i] > 0 else "SELL",
            )
            for i in range(len(ts))
        ]
        if batch_ticks:
            yield batch_ticks


def load_ticks_csv(
    path: str,
    start: str = None,
    end: str = None,
    start_time: str = None,
    end_time: str = None,
) -> list[Tick]:
    """
    Two formats accepted:

    A) Simple:  ts_ms, price, qty, side
    B) Binance aggTrades: agg_id, price, qty, first_id, last_id, ts_ms, is_buyer_maker
       is_buyer_maker=True → aggressive SELL (buyer is maker = seller is taker)
    """
    ticks = []
    count = 0
    sd_ms, ed_ms = _range_ms(start, end, start_time, end_time)
    skipped_out_of_range = 0

    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith(("#", "ts", "agg", "T")):
                continue
            row = [r.strip() for r in row]
            try:
                if len(row) >= 7:                        # Binance aggTrades
                    ts_ms = int(row[5])
                    price = float(row[1])
                    qty   = float(row[2])
                    side  = "sell" if row[6].lower() in ("true", "1") else "buy"
                elif len(row) >= 4:                      # simple format
                    ts_ms = int(row[0])
                    price = float(row[1])
                    qty   = float(row[2])
                    side  = row[3].lower()
                else:
                    continue
                if sd_ms is not None and ts_ms < sd_ms:
                    skipped_out_of_range += 1
                    continue
                if ed_ms is not None and ts_ms > ed_ms:
                    skipped_out_of_range += 1
                    continue
                ticks.append(Tick(ts=ts_ms, price=price, qty=qty, side=side))
                count += 1
                if count % 500_000 == 0:
                    print(f"       Loaded {count:,} ticks…", flush=True)
            except (ValueError, IndexError):
                continue
    if skipped_out_of_range:
        print(f"      ✓ {count:,} ticks loaded ({skipped_out_of_range:,} skipped by date range)", flush=True)
    else:
        print(f"      ✓ {count:,} total ticks loaded", flush=True)
    return sorted(ticks, key=lambda t: t.ts)


def load_ticks_dir(
    directory: str,
    start: str = None,
    end: str = None,
    start_time: str = None,
    end_time: str = None,
) -> list[Tick]:
    sd = datetime.strptime(start, "%Y-%m-%d") if start else None
    ed = datetime.strptime(end,   "%Y-%m-%d") if end   else None
    all_ticks: list[Tick] = []
    for f in sorted(Path(directory).glob("**/*.csv")):
        try:
            fdt = datetime.strptime(f.stem[:8], "%Y%m%d")
            if sd and fdt < sd: continue
            if ed and fdt > ed: continue
        except ValueError:
            pass
        all_ticks.extend(load_ticks_csv(str(f), start, end, start_time, end_time))
    return sorted(all_ticks, key=lambda t: t.ts)


def filter_ticks_by_range(ticks: list[Tick], start: str = None, end: str = None,
                          start_time: str = None, end_time: str = None) -> list[Tick]:
    """
    Filter ticks by date range and optional intraday time window.

    Parameters
    ----------
    ticks : list[Tick]
        All ticks to filter
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD, inclusive)
    start_time : str
        Start time (HH:MM, e.g. "09:30")
    end_time : str
        End time (HH:MM, e.g. "16:00")

    Returns
    -------
    list[Tick]
        Filtered ticks
    """
    if not ticks:
        return []

    sd_ms = None
    ed_ms = None

    if start:
        sd = datetime.strptime(start, "%Y-%m-%d")
        if start_time:
            h, m = map(int, start_time.split(":"))
            sd = sd.replace(hour=h, minute=m, second=0, microsecond=0)
        sd_ms = int(sd.timestamp() * 1000)

    if end:
        ed = datetime.strptime(end, "%Y-%m-%d")
        if end_time:
            h, m = map(int, end_time.split(":"))
            ed = ed.replace(hour=h, minute=m, second=59, microsecond=999999)
        else:
            ed = ed.replace(hour=23, minute=59, second=59, microsecond=999999)
        ed_ms = int(ed.timestamp() * 1000)

    filtered = []
    for t in ticks:
        if sd_ms and t.ts < sd_ms:
            continue
        if ed_ms and t.ts > ed_ms:
            continue
        filtered.append(t)

    return filtered


def compress_ticks(ticks: list[Tick], bucket_ms: int) -> list[Tick]:
    """
    Compress dense tape by aggregating ticks into (time-bucket, side) events.
    Price is VWAP, quantity is summed.
    """
    if bucket_ms <= 0 or not ticks:
        return ticks

    ts = np.fromiter((t.ts for t in ticks), dtype=np.int64, count=len(ticks))
    px = np.fromiter((t.price for t in ticks), dtype=np.float64, count=len(ticks))
    qty = np.fromiter((t.qty for t in ticks), dtype=np.float64, count=len(ticks))
    side_code = np.fromiter(
        (1 if t.side.upper().startswith("B") else -1 for t in ticks),
        dtype=np.int8,
        count=len(ticks),
    )
    c_ts, c_px, c_qty, c_side = _compress_ticks_numba(ts, px, qty, side_code, bucket_ms)
    return [
        Tick(
            ts=int(c_ts[i]),
            price=float(c_px[i]),
            qty=float(c_qty[i]),
            side="buy" if c_side[i] > 0 else "sell",
        )
        for i in range(len(c_ts))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TICK → CANDLE PIPELINE
# Uses orderflow_indicators.py when present, built-in fallback otherwise.
# ══════════════════════════════════════════════════════════════════════════════
def ticks_to_candles_streaming(csv_path: str, cfg: BacktestConfig, 
                               start: str = None, end: str = None,
                               start_time: str = None, end_time: str = None):
    """
    Stream ticks from CSV, build candles in-memory without storing all ticks.
    Yields candles as they complete.
    Memory: O(candle_width) instead of O(all_ticks)
    """
    if not Path(csv_path).exists():
        return
    
    tf_ms = cfg.timeframe_minutes * 60_000
    buckets = defaultdict(list)
    flows, closes = [], []
    count = 0
    last_open_ts = None
    
    # Parse date ranges
    sd_ms = None; ed_ms = None
    if start:
        sd = datetime.strptime(start, "%Y-%m-%d")
        if start_time:
            h, m = map(int, start_time.split(":"))
            sd = sd.replace(hour=h, minute=m, second=0)
        sd_ms = int(sd.timestamp() * 1000)
    if end:
        ed = datetime.strptime(end, "%Y-%m-%d")
        if end_time:
            h, m = map(int, end_time.split(":"))
            ed = ed.replace(hour=h, minute=m, second=59, microsecond=999999)
        else:
            ed = ed.replace(hour=23, minute=59, second=59)
        ed_ms = int(ed.timestamp() * 1000)
    
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith(("#", "ts", "agg", "T")):
                continue
            try:
                row = [r.strip() for r in row]
                if len(row) >= 7:  # Binance aggTrades
                    ts_ms = int(row[5])
                    price = float(row[1])
                    qty = float(row[2])
                    side = "sell" if row[6].lower() in ("true", "1") else "buy"
                elif len(row) >= 4:
                    ts_ms = int(row[0])
                    price = float(row[1])
                    qty = float(row[2])
                    side = row[3].lower()
                else:
                    continue
                
                # Filter by date/time
                if sd_ms and ts_ms < sd_ms: continue
                if ed_ms and ts_ms > ed_ms: continue
                
                count += 1
                if count % 1_000_000 == 0:
                    print(f"        Streamed {count:,} ticks, {len(flows)} candles built…", flush=True)
                
                # Bucket into candles
                open_ts = (ts_ms // tf_ms) * tf_ms
                if last_open_ts and open_ts != last_open_ts:
                    # Candle boundary crossed — emit previous
                    yield _build_candle(buckets[last_open_ts], last_open_ts, tf_ms, cfg, flows, closes)
                    del buckets[last_open_ts]
                
                buckets[open_ts].append({"ts": ts_ms, "price": price, "qty": qty, "side": side})
                last_open_ts = open_ts
            except (ValueError, IndexError):
                continue
    
    # Emit remaining
    if last_open_ts and last_open_ts in buckets:
        yield _build_candle(buckets[last_open_ts], last_open_ts, tf_ms, cfg, flows, closes)
    
    print(f"      ✓ {count:,} total ticks streamed, {len(flows)} candles built", flush=True)


@njit(cache=True)
def _build_candle_numba_core(prices, qtys, side_codes, large_trade_pct):
    n = prices.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bv = 0.0
    sv = 0.0
    max_p = prices[0]
    min_p = prices[0]
    for i in range(n):
        q = qtys[i]
        p = prices[i]
        if side_codes[i] > 0:
            bv += q
        else:
            sv += q
        if p > max_p:
            max_p = p
        if p < min_p:
            min_p = p
    tv = bv + sv
    delta = bv - sv

    sorted_qtys = np.sort(qtys)
    idx = int(n * large_trade_pct)
    if idx >= n:
        idx = n - 1
    thresh = sorted_qtys[idx] if n > 0 else 0.0
    lb = 0.0
    ls = 0.0
    for i in range(n):
        q = qtys[i]
        if q > thresh:
            if side_codes[i] > 0:
                lb += q
            else:
                ls += q
    close_p = prices[n - 1]
    return bv, sv, tv, delta, lb, ls, max_p, min_p, close_p


def _build_candle(tl, open_ts, tf_ms, cfg, flows_list, closes_list):
    """Build single candle from tick list, update running lists."""
    if not tl:
        return None
    prices = np.array([t["price"] for t in tl], dtype=np.float64)
    qtys = np.array([t["qty"] for t in tl], dtype=np.float64)
    side_codes = np.array([1 if t["side"].upper() == "BUY" else -1 for t in tl], dtype=np.int8)
    bv, sv, tv, delta, lb, ls, max_p, min_p, cp = _build_candle_numba_core(
        prices, qtys, side_codes, float(cfg.large_trade_pct)
    )
    
    # Build CandleFlow
    f = CandleFlow(
        open_ts=open_ts, close_ts=open_ts+tf_ms,
        buy_vol=bv, sell_vol=sv, delta=delta, total_vol=tv,
        large_buy_vol=lb, large_sell_vol=ls,
        max_price=max_p,
        min_price=min_p,
        close_price=cp
    ) if _INDICATORS_AVAILABLE else type('F', (), {
        'open_ts': open_ts, 'close_ts': open_ts+tf_ms,
        'buy_vol': bv, 'sell_vol': sv, 'delta': delta, 'total_vol': tv,
        'large_buy_vol': lb, 'large_sell_vol': ls,
        'max_price': max_p,
        'min_price': min_p,
        'close_price': cp,
        'cvd': sum(f.delta for f in flows_list) + delta if flows_list else delta,
        'imbalance': (bv - sv) / tv if tv > 0 else 0,
        'absorption': (lb - ls) / tv if tv > 0 else 0,
        'stacked_imb': 0, 'delta_div': 0, 'ob_imbalance': 0,
        'bid_vol': bv, 'ask_vol': sv
    })()
    
    flows_list.append(f)
    closes_list.append(f.close_price)
    return f


def _fallback_compute(buckets, bucket_times, tf_ms, cfg):
    """Minimal inline computation when orderflow_indicators isn't importable."""

    @dataclass
    class _Flow:
        open_ts: int;  close_ts: int
        buy_vol: float = 0.0;  sell_vol: float = 0.0
        delta: float = 0.0;    total_vol: float = 0.0
        buy_trades: int = 0;   sell_trades: int = 0
        large_buy_vol: float = 0.0;  large_sell_vol: float = 0.0
        max_price: float = 0.0;  min_price: float = 0.0;  close_price: float = 0.0
        cvd: float = 0.0;  bid_vol: float = 0.0;  ask_vol: float = 0.0
        ob_imbalance: float = 0.0;  imbalance: float = 0.0;  absorption: float = 0.0
        stacked_imb: float = 0.0;   delta_div: float = 0.0

    flows, closes = [], []
    cvd_run = prev_stack = 0.0
    dw = cfg.divergence_window

    for i, open_ts in enumerate(bucket_times):
        tl = buckets[open_ts]
        bv  = sum(t["qty"] for t in tl if t["side"].upper() == "BUY")
        sv  = sum(t["qty"] for t in tl if t["side"].upper() == "SELL")
        tv  = bv + sv;  delta = bv - sv;  cvd_run += delta
        qtys = sorted(t["qty"] for t in tl)
        thresh = qtys[int(len(qtys) * cfg.large_trade_pct)] if qtys else 0
        lb = sum(t["qty"] for t in tl if t["side"].upper() == "BUY"  and t["qty"] > thresh)
        ls = sum(t["qty"] for t in tl if t["side"].upper() == "SELL" and t["qty"] > thresh)
        prices = [t["price"] for t in tl]
        cp = tl[-1]["price"]
        imb = delta / tv if tv > 0 else 0.0
        if imb > 0 and prev_stack > 0:   stack = prev_stack + 1
        elif imb < 0 and prev_stack < 0: stack = prev_stack - 1
        elif imb > 0: stack = 1.0
        elif imb < 0: stack = -1.0
        else: stack = 0.0
        prev_stack = stack
        abs_v = (lb - ls) / tv if tv > 1e-9 else 0.0
        div = 0.0
        if i >= dw:
            pc = cp - closes[i - dw]
            dc = delta - flows[i - dw].delta
            if pc > 0 and dc < 0: div = 1.0
            elif pc < 0 and dc > 0: div = -1.0
        f = _Flow(open_ts=open_ts, close_ts=open_ts+tf_ms, buy_vol=bv, sell_vol=sv,
                  delta=delta, total_vol=tv, large_buy_vol=lb, large_sell_vol=ls,
                  max_price=max(prices) if prices else cp, min_price=min(prices) if prices else cp,
                  close_price=cp, cvd=cvd_run, bid_vol=bv, ask_vol=sv,
                  imbalance=imb, absorption=abs_v, stacked_imb=stack, delta_div=div)
        flows.append(f); closes.append(cp)
    return flows, closes


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC TICK GENERATOR (demo mode)
# ══════════════════════════════════════════════════════════════════════════════
def generate_demo_ticks(n_candles: int = 2000, tf_minutes: int = 5,
                        seed: int = 42) -> list[Tick]:
    """Realistic BTC-like tick data — ~50 ticks per candle, trending regimes."""
    rng   = random.Random(seed)
    tf_ms = tf_minutes * 60_000
    ticks = []
    price = 43_500.0
    regime, rc = 1, 0
    t0_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    for ci in range(n_candles):
        rc += 1
        if rc > rng.randint(25, 100):
            regime = -regime; rc = 0
        price *= 1 + (regime * 0.0004 + rng.gauss(0, 0.003))
        open_ms = t0_ms + ci * tf_ms
        for _ in range(rng.randint(30, 80)):
            ts  = open_ms + rng.randint(0, tf_ms - 1)
            qty = rng.lognormvariate(-3.5, 1.2)
            p   = price * (1 + rng.gauss(0, 0.0003))
            s   = "buy" if rng.random() < (0.58 if regime == 1 else 0.42) else "sell"
            ticks.append(Tick(ts=ts, price=round(p, 2), qty=round(qty, 6), side=s))

    return sorted(ticks, key=lambda t: t.ts)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE STRATEGY WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
class LiveStrategyBacktester:
    """
    Wrapper that feeds historical data to the actual live strategy.
    
    This ensures backtest uses IDENTICAL logic to live trading.
    """
    
    def __init__(self, config_path: str, initial_capital: float = 10_000.0):
        self.stack = load_orderflow_config(Path(config_path))
        self.strategy_config = orderflow_strategy_config_from_stack(self.stack)
        
        # DEBUG: Log actual VP config being loaded
        vp_cfg = self.strategy_config.vp_config or {}
        hm_cfg = self.strategy_config.heatmap_config or {}
        print(f"=== LOADED VP CONFIG: {vp_cfg} ===")
        print(f"=== LOADED HEATMAP CONFIG: {hm_cfg} ===")
        print(f"=== HEATMAP stop_buffer_bps: {hm_cfg.get('stop_buffer_bps', 'NOT_FOUND')} ===")
        
        self.strategy = OrderflowStrategy(self.strategy_config)

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: list[Trade] = []
        self.trade_counter = 0
        self.position = None

        tf = self.strategy_config.timeframe
        if tf.endswith("m"):
            tf_minutes = int(tf[:-1])
        elif tf.endswith("h"):
            tf_minutes = int(tf[:-1]) * 60
        else:
            tf_minutes = 5
        self._tf_ms = tf_minutes * 60_000

        self._entry_stop_price: float | None = None
        self._entry_target_price: float | None = None
        self._trailing_peak: float = 0.0
        self._accept_level_price: float | None = None
        self._accept_band_bps: float | None = None
        self._accept_fail_ms: int = 0
        self._accept_last_ts_ms: int | None = None

    def _execution_costs(self) -> tuple[float, float]:
        """Return (fee_rate, slippage_rate) for synthetic fills."""
        fee_rate = float(getattr(self.strategy_config, "backtest_fee_rate", 0.0004))
        slippage = float(getattr(self.strategy_config, "backtest_slippage_rate", 0.0002))
        return max(0.0, fee_rate), max(0.0, slippage)

    def _sanitize_brackets(self, side: str, fill_price: float) -> None:
        """
        Ensure stop/target are structurally correct relative to executed entry.
        If levels are inverted/too tight, disable that level to avoid fake exits.
        """
        min_edge = 1e-6
        fee_rate, slippage = self._execution_costs()
        # Require target edge to beat synthetic round-trip friction.
        min_target_move = 2.0 * (fee_rate + slippage) + 1e-6
        stop = self._entry_stop_price
        target = self._entry_target_price

        if side == "BUY":
            if stop is not None and stop >= fill_price * (1.0 - min_edge):
                stop = None
            if target is not None and target <= fill_price * (1.0 + min_target_move):
                target = None
        else:
            if stop is not None and stop <= fill_price * (1.0 + min_edge):
                stop = None
            if target is not None and target >= fill_price * (1.0 - min_target_move):
                target = None

        self._entry_stop_price = stop
        self._entry_target_price = target

    def _entry_geometry_is_valid(self, side: str, fill_price: float) -> bool:
        """
        Enforce structural quality gates before opening the position.
        Skip entries with poor wall geometry (too much risk or weak reward/risk).
        """
        stop = self._entry_stop_price
        target = self._entry_target_price
        if stop is None or target is None:
            return False  # Reject entries without structural levels

        if side == "BUY":
            risk = (fill_price - stop) / fill_price
            reward = (target - fill_price) / fill_price
        else:
            risk = (stop - fill_price) / fill_price
            reward = (fill_price - target) / fill_price

        if risk <= 0.0 or reward <= 0.0:
            return False

        rr = reward / risk
        max_stop_bps = float(getattr(self.strategy_config, "max_structural_stop_bps", 35.0))
        min_rr = float(getattr(self.strategy_config, "min_structural_rr", 1.2))
        risk_bps = risk * 10_000.0
        return risk_bps <= max_stop_bps and rr >= min_rr

    def _entry_risk_metrics(self, side: str, fill_price: float) -> dict:
        """Compute stop/target geometry diagnostics for this entry."""
        stop = self._entry_stop_price
        target = self._entry_target_price
        if stop is None or target is None:
            return {
                "has_structural_levels": False,
                "stop_bps": None,
                "target_bps": None,
                "rr": None,
            }
        if side == "BUY":
            risk = (fill_price - stop) / fill_price
            reward = (target - fill_price) / fill_price
        else:
            risk = (stop - fill_price) / fill_price
            reward = (fill_price - target) / fill_price
        rr = (reward / risk) if risk > 0 else None
        return {
            "has_structural_levels": True,
            "stop_bps": risk * 10_000.0,
            "target_bps": reward * 10_000.0,
            "rr": rr,
        }

    def run_backtest(
        self,
        tick_batches: Iterable[list[Tick]],
        *,
        total_ticks: int | None = None,
        fast_mode: bool = False,
        heatmap_stride: int = 1,
        vp_stride: int = 1,
    ) -> tuple[list[Trade], int]:
        from nautilus_trader.model.enums import OrderSide

        tf_ms = self._tf_ms
        if total_ticks:
            print(
                f"   Running {total_ticks:,} ticks  | candle-boundary eval "
                f"(~{86_400_000 // tf_ms} evals/day)\n"
            )

        completed_candles = 0
        current_candle_open = None
        last_session = None
        last_session_min = None
        started_at = last_heartbeat = perf_counter()
        processed = 0
        last_price = None
        last_ts = None

        if fast_mode:
            heatmap_stride = max(heatmap_stride, 8)
            vp_stride = max(vp_stride, 4)

        # Backtest has no true order book deltas; use short-horizon tape imbalance proxy.
        ob_window = int(getattr(self.strategy_config, "ob_proxy_window_ticks", 200))
        ob_window = max(20, ob_window)
        signed_qty_window: deque[float] = deque(maxlen=ob_window)
        abs_qty_window: deque[float] = deque(maxlen=ob_window)

        for batch in tick_batches:
            if not batch:
                continue
            for t in batch:
                side = t.side.upper()
                if side not in {"BUY", "SELL"}:
                    continue
                processed += 1
                last_price = t.price
                last_ts = t.ts

                self.strategy._engine.add_tick(t.ts, t.price, t.qty, side)
                # ALWAYS feed heatmap every tick — structural stops require full tape
                self.strategy._heatmap_engine.add_trade(t.price, t.qty)
                if processed % vp_stride == 0:
                    self.strategy._vp_engine.add_trade(t.price, t.qty)
                signed = t.qty if side == "BUY" else -t.qty
                signed_qty_window.append(signed)
                abs_qty_window.append(abs(t.qty))
                denom = sum(abs_qty_window)
                ob_proxy = (sum(signed_qty_window) / denom) if denom > 1e-9 else 0.0
                self.strategy._engine.set_orderbook_imbalance_value(ob_proxy)

                this_candle = (t.ts // tf_ms) * tf_ms
                if this_candle != current_candle_open:
                    if current_candle_open is not None:
                        completed_candles += 1
                        eval_ts = t.ts - 1
                        snap = self.strategy._engine.compute_snapshot(now_ms=eval_ts)

                        if snap is not None:
                            px = snap.ltf.close_price
                            self.strategy._heatmap = (
                                self.strategy._heatmap_engine.compute_snapshot(px)
                                if self.strategy._heatmap_engine.is_warm else None
                            )
                            self.strategy._vp = (
                                self.strategy._vp_engine.compute_snapshot(px)
                                if self.strategy._vp_engine.is_warm else None
                            )

                            htf_candles = self.strategy._engine.completed_candles(
                                getattr(self.strategy.config, "htf_timeframe", "1h")
                            )
                            if htf_candles and snap.htf:
                                self.strategy._structure = self.strategy._structure_engine.update(
                                    htf_candles, snap.htf.close_price
                                )

                            minute_key = eval_ts // 60_000
                            if last_session_min != minute_key:
                                last_session = self.strategy._session_filter.current_session(
                                    datetime.fromtimestamp(eval_ts / 1000.0, timezone.utc)
                                )
                                last_session_min = minute_key

                            if self.position is None:
                                self._check_entry_signals(snap, last_session)
                    current_candle_open = this_candle

                if self.position is not None:
                    pos_side = (
                        OrderSide.BUY if self.position["direction"] == "long" else OrderSide.SELL
                    )
                    self._check_exit_signals(t.price, t.ts, pos_side)

                now = perf_counter()
                if (now - last_heartbeat) >= 5.0:
                    elapsed = max(now - started_at, 1e-9)
                    tps = processed / elapsed
                    if total_ticks:
                        eta_m = (total_ticks - processed) / tps / 60 if tps > 0 else float("inf")
                        print(
                            f"     {processed/total_ticks*100:5.1f}% | {processed:,}/{total_ticks:,} | "
                            f"{tps:,.0f} ticks/s | {completed_candles} candles | ETA {eta_m:.1f}m",
                            flush=True,
                        )
                    else:
                        print(
                            f"     {processed:,} ticks | {tps:,.0f} ticks/s | {completed_candles} candles",
                            flush=True,
                        )
                    last_heartbeat = now

        if self.position and last_price is not None and last_ts is not None:
            self._close_position("end_of_data", last_price, last_ts)

        return self.trades, completed_candles
    
    def _filter_flows_by_date(self, flows: list[CandleFlow], start_date: str = None, end_date: str = None) -> list[CandleFlow]:
        """
        Filter candle flows by date range.
        """
        if not start_date and not end_date:
            return flows
            
        start_ts = None
        end_ts = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_ts = int(start_dt.timestamp() * 1000)
            
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            end_ts = int(end_dt.timestamp() * 1000)
            
        filtered_flows = []
        for flow in flows:
            if start_ts and flow.open_ts < start_ts:
                continue
            if end_ts and flow.open_ts > end_ts:
                continue
            filtered_flows.append(flow)
            
        return filtered_flows
    
    def _check_entry_signals(self, snap, session):
        """
        Check if live strategy signals entry using actual signal modules.
        """
        # REQUIRE HEATMAP WARMUP before processing any signals
        if not self.strategy._heatmap_engine.is_warm:
            return  # Skip entry until heatmap is warm (structural stops must be reliable)
        
        long_signal = self.strategy._evaluate_direction(snap, session, is_long=True)
        short_signal = self.strategy._evaluate_direction(snap, session, is_long=False)

        if long_signal and short_signal:
            if long_signal.confidence > short_signal.confidence:
                signal = long_signal
            elif short_signal.confidence > long_signal.confidence:
                signal = short_signal
            else:
                return
        else:
            signal = long_signal or short_signal

        if signal is None:
            return

        ml_confidence = self.strategy._inference_hook.predict(
            self.strategy._build_feature_row(snap, session, signal),
        )
        if ml_confidence <= 0:
            return
        combined_conf = float(ml_confidence) * float(getattr(signal, "confidence", 1.0))
        if combined_conf <= 0:
            return

        flow = snap.ltf.flow
        details = [signal.label, f"conf={combined_conf:.2f}"]
        self._open_position(flow, snap.ltf.close_price, "+".join(details), signal.side.name)

    def _check_exit_signals(self, current_price: float, ts_ms: int, position_side):
        """
        Check if live strategy signals exit using actual exit logic.
        Matches live strategy sequential if logic (not elif chain).
        """
        if not self.position:
            return
            
        entry_price = self.position['entry_price']
        direction = 1 if position_side.name == 'BUY' else -1
        pnl_pct = direction * (current_price - entry_price) / entry_price
        self.position["best_pnl_pct"] = max(self.position["best_pnl_pct"], pnl_pct * 100.0)
        self.position["worst_pnl_pct"] = min(self.position["worst_pnl_pct"], pnl_pct * 100.0)
        held_secs = (ts_ms / 1000.0 - self.position['entry_time'].timestamp())

        # 1. Hard stop (emergency backstop) - always checked first
        stop_dist = self.strategy_config.stoploss_pct
        if pnl_pct <= -stop_dist:
            # Use trigger stop price (not overshot tick) to avoid exaggerated losses.
            if position_side.name == 'BUY':
                stop_px = entry_price * (1.0 - stop_dist)
            else:
                stop_px = entry_price * (1.0 + stop_dist)
            self._close_position("hard_stop", stop_px, ts_ms)
            return

        # 2. Acceptance-failure exit (breakout entries)
        if self._accept_level_price is not None and self._accept_band_bps is not None:
            band = float(self._accept_band_bps) / 10_000.0
            if position_side.name == "BUY":
                failed = current_price < (self._accept_level_price * (1.0 - band))
            else:
                failed = current_price > (self._accept_level_price * (1.0 + band))

            if self._accept_last_ts_ms is None:
                self._accept_last_ts_ms = ts_ms
            dt = max(0, ts_ms - self._accept_last_ts_ms)
            self._accept_last_ts_ms = ts_ms
            if failed:
                self._accept_fail_ms += dt
                fail_secs = float(getattr(self.strategy_config, "acceptance_failure_secs", 60.0))
                if self._accept_fail_ms >= int(fail_secs * 1000):
                    self._close_position("acceptance_fail", current_price, ts_ms)
                    return
            else:
                self._accept_fail_ms = 0

        # 3. Structural stop (wall break)
        if self._entry_stop_price is not None:
            if position_side.name == 'BUY' and current_price <= self._entry_stop_price:
                # Fill at configured structural stop trigger to reduce bucket overshoot bias.
                self._close_position("wall_break_stop", self._entry_stop_price, ts_ms)
                return
            if position_side.name == 'SELL' and current_price >= self._entry_stop_price:
                # Fill at configured structural stop trigger to reduce bucket overshoot bias.
                self._close_position("wall_break_stop", self._entry_stop_price, ts_ms)
                return

        # 4. Min hold time - suppress exits before this
        min_hold_secs = getattr(self.strategy_config, 'min_hold_secs', 15.0)
        if min_hold_secs > 0:
            if held_secs < min_hold_secs:
                return

        # 5. Target - now reachable
        if self._entry_target_price is not None:
            if position_side.name == 'BUY' and current_price >= self._entry_target_price:
                # Acceptance trades: don't scalp the first objective, trail a runner.
                if self.position and "acceptance" in str(self.position.get("entry_reason", "")):
                    self._entry_target_price = None
                    # activate trailing immediately at objective hit
                    self._trailing_peak = max(self._trailing_peak, current_price)
                    return
                self._close_position("wall_target_hit", current_price, ts_ms)
                return
            if position_side.name == 'SELL' and current_price <= self._entry_target_price:
                if self.position and "acceptance" in str(self.position.get("entry_reason", "")):
                    self._entry_target_price = None
                    self._trailing_peak = min(self._trailing_peak, current_price)
                    return
                self._close_position("wall_target_hit", current_price, ts_ms)
                return

        # 6. Trailing stop - now reachable
        trailing_trigger = getattr(self.strategy_config, 'trailing_trigger_pct', 0.012)
        trailing_offset = getattr(self.strategy_config, 'trailing_offset_pct', 0.006)
        if pnl_pct >= trailing_trigger:
            if position_side.name == 'BUY':
                self._trailing_peak = max(self._trailing_peak, current_price)
                trail_dd = (current_price - self._trailing_peak) / self._trailing_peak
            else:
                self._trailing_peak = min(self._trailing_peak, current_price)
                trail_dd = (self._trailing_peak - current_price) / self._trailing_peak
            if trail_dd <= -trailing_offset:
                self._close_position("trailing_stop", current_price, ts_ms)
                return

        # 7. Time stop - now reachable
        max_secs = getattr(self.strategy_config, 'max_time_in_trade_secs', None)
        if max_secs:
            open_secs = (ts_ms / 1000.0 - self.position['entry_time'].timestamp())
            if open_secs >= max_secs:
                self._close_position("time_stop", current_price, ts_ms)
                return
                    
    def _open_position(self, flow: CandleFlow, entry_price: float, reason: str, side: str):
        """
        Open a position using the same sizing logic as live strategy.
        """
        if self.current_capital <= 0:
            return

        # Scale position by confidence embedded in entry_reason (conf=...).
        conf_mult = 1.0
        if "conf=" in reason:
            try:
                conf_mult = float(reason.split("conf=")[-1])
            except ValueError:
                conf_mult = 1.0
        conf_mult = max(0.0, min(1.0, conf_mult))

        # Setup-specific controls: short rejection setup gets stricter filters and reduced size.
        is_short_rejection = reason.startswith("poc_rejection_short")
        if is_short_rejection:
            short_min_conf = float(getattr(self.strategy_config, "short_rejection_min_confidence", 1.05))
            try:
                conf_token = reason.split("conf=")[-1]
                conf_val = float(conf_token)
            except (ValueError, IndexError):
                conf_val = 0.0
            if conf_val < short_min_conf:
                return
            conf_mult *= float(getattr(self.strategy_config, "short_rejection_size_mult", 0.6))
            conf_mult = max(0.0, min(1.0, conf_mult))
            
        # Simulate fees and slippage (configurable in execution.* for backtests).
        fee_rate, slippage = self._execution_costs()

        if side == 'BUY':
            fill_price = entry_price * (1 + slippage)
        else:
            fill_price = entry_price * (1 - slippage)
        
        # Store structural stop and target prices from HEATMAP (not VP)
        # Use HEATMAP for stops (real structural support/resistance walls)
        # Use VP only for acceptance entry targets (POC/VAH/VAL objectives)
        self._entry_stop_price = None
        self._entry_target_price = None
        self._trailing_peak = entry_price
        
        if self.strategy._heatmap is not None:
            if side == 'BUY':
                # Structural stop from heatmap wall
                self._entry_stop_price = self.strategy._heatmap.long_stop_price
                # Target: use VP for acceptance trades, heatmap otherwise
                if "acceptance" in reason and self.strategy._vp is not None and self.strategy._vp.is_valid:
                    candidates = [
                        self.strategy._vp.long_target_price,
                        self.strategy._vp.long_travel_target_price,
                        self.strategy._vp.vah_price,
                    ]
                    candidates = [c for c in candidates if c is not None and c > fill_price]
                    self._entry_target_price = max(candidates) if candidates else self.strategy._heatmap.long_target_price
                else:
                    self._entry_target_price = self.strategy._heatmap.long_target_price
            else:
                # Structural stop from heatmap wall
                self._entry_stop_price = self.strategy._heatmap.short_stop_price
                # Target: use VP for acceptance trades, heatmap otherwise
                if "acceptance" in reason and self.strategy._vp is not None and self.strategy._vp.is_valid:
                    candidates = [
                        self.strategy._vp.short_target_price,
                        self.strategy._vp.short_travel_target_price,
                        self.strategy._vp.val_price,
                    ]
                    candidates = [c for c in candidates if c is not None and c < fill_price]
                    self._entry_target_price = min(candidates) if candidates else self.strategy._heatmap.short_target_price
                else:
                    self._entry_target_price = self.strategy._heatmap.short_target_price

        self._sanitize_brackets(side, fill_price)
        self._accept_level_price = None
        self._accept_band_bps = None
        self._accept_fail_ms = 0
        self._accept_last_ts_ms = None

        # Configure acceptance boundary from the VP snapshot at entry time.
        vp = self.strategy._vp
        vp_cfg = self.strategy_config.vp_config or {}
        poc_band = float(vp_cfg.get("poc_band_bps", 8.0))
        va_band = float(vp_cfg.get("va_band_bps", 10.0))
        if vp is not None and vp.is_valid:
            if reason.startswith("poc_acceptance_retest_"):
                self._accept_level_price = vp.poc_price
                self._accept_band_bps = poc_band
            elif reason.startswith("vah_acceptance_long"):
                self._accept_level_price = vp.vah_price
                self._accept_band_bps = va_band
            elif reason.startswith("val_acceptance_short"):
                self._accept_level_price = vp.val_price
                self._accept_band_bps = va_band

        # Risk-based sizing (matches live intent): fixed risk fraction per trade.
        if not self._entry_stop_price or fill_price <= 0:
            return

        stop_bps = abs(fill_price - self._entry_stop_price) / fill_price * 10_000.0
        min_stop_bps = float(getattr(self.strategy_config, "min_stop_bps", 0.0))
        max_stop_bps = float(getattr(self.strategy_config, "max_stop_bps", 10_000.0))
        if stop_bps < min_stop_bps or stop_bps > max_stop_bps:
            return

        risk_per_trade = float(getattr(self.strategy_config, "risk_per_trade_pct", 0.0))
        risk_amount = self.current_capital * max(0.0, risk_per_trade) * conf_mult
        if risk_amount <= 0:
            return

        stop_dist = abs(fill_price - self._entry_stop_price)
        if stop_dist <= 0:
            return

        qty_risk = risk_amount / stop_dist

        max_fraction = float(getattr(self.strategy_config, "max_position_fraction", 0.0))
        max_notional = getattr(self.strategy_config, "max_notional_usdt", None)
        notional_cap = self.current_capital * max(0.0, min(1.0, max_fraction))
        if max_notional is not None:
            try:
                notional_cap = min(notional_cap, float(max_notional))
            except (TypeError, ValueError):
                pass

        # Include entry fee in notional cap so we don't overspend cash.
        qty_cap = notional_cap / (fill_price * (1.0 + fee_rate)) if notional_cap > 0 else 0.0
        qty = min(qty_risk, qty_cap) if qty_cap > 0 else qty_risk
        if qty <= 0:
            return

        entry_fee = (fill_price * qty) * fee_rate
        capital_to_use = fill_price * qty + entry_fee
        if capital_to_use < 10:  # Minimum trade size
            return

        # Live strategy has its own geometry validation now
        risk_metrics = self._entry_risk_metrics(side, fill_price)
        
        # DEBUG: Log stop distances and VP state
        if self._entry_stop_price:
            stop_dist_bps = abs(fill_price - self._entry_stop_price) / fill_price * 10000
            vp_state = "VP" if self.strategy._vp and self.strategy._vp.is_valid else "HEATMAP" if self.strategy._heatmap else "NONE"
            print(f"[DEBUG] {side} entry @ {fill_price:.2f}, stop @ {self._entry_stop_price:.2f}, distance: {stop_dist_bps:.1f} bps, source: {vp_state}")
            if self.strategy._vp:
                print(f"[DEBUG] VP buckets: {self.strategy._vp_engine.bucket_count}, total_vol: {self.strategy._vp_engine.total_volume:.0f}")
        else:
            print(f"[DEBUG] {side} entry @ {fill_price:.2f}, NO STOP PRICE!")
        
        self.position = {
            'entry_price': fill_price,
            'qty': qty,
            'capital_used': capital_to_use,
            'direction': 'long' if side == 'BUY' else 'short',
            'entry_time': datetime.fromtimestamp(flow.open_ts / 1000, timezone.utc),
            'entry_reason': reason,
            'entry_side': side,
            'best_pnl_pct': 0.0,
            'worst_pnl_pct': 0.0,
            'entry_diagnostics': {
                "stop_price": self._entry_stop_price,
                "target_price": self._entry_target_price,
                "stop_bps": risk_metrics["stop_bps"],
                "target_bps": risk_metrics["target_bps"],
                "rr": risk_metrics["rr"],
                "has_structural_levels": risk_metrics["has_structural_levels"],
                "is_short_rejection": is_short_rejection,
            },
        }
        
        self.current_capital -= capital_to_use
        
    def _close_position(self, reason: str, exit_price: float, exit_ts: int):
        """
        Close current position.
        """
        if not self.position:
            return
            
        # Simulate fees and slippage (configurable in execution.* for backtests).
        fee_rate, slippage = self._execution_costs()
        
        # Apply slippage in correct direction based on position
        if self.position['direction'] == 'long':
            # Long: exit with SELL order, slippage hurts us (worse price)
            fill_price = exit_price * (1 - slippage)
        else:  # short
            # Short: exit with BUY order, slippage hurts us (worse price)
            fill_price = exit_price * (1 + slippage)
        fee = fill_price * self.position['qty'] * fee_rate
        
        # Fix PnL calculation for short vs long positions
        if self.position['direction'] == 'long':
            pnl = (fill_price - self.position['entry_price']) * self.position['qty'] - fee
        else:  # short
            pnl = (self.position['entry_price'] - fill_price) * self.position['qty'] - fee
        pnl_pct = pnl / self.position['capital_used'] * 100
        
        self.current_capital += self.position['capital_used'] + pnl
        
        # Calculate duration
        entry_time = self.position['entry_time']
        exit_time_dt = datetime.fromtimestamp(exit_ts / 1000, timezone.utc)
        duration_mins = (exit_time_dt - entry_time).total_seconds() / 60
        
        # Record trade with actual entry and exit reasons
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            entry_time=entry_time,
            exit_time=exit_time_dt,
            entry_price=self.position['entry_price'],
            exit_price=fill_price,
            qty=self.position['qty'],
            capital_used=self.position['capital_used'],
            entry_reason=self.position.get('entry_reason', 'entry'),
            exit_reason=reason,
            pnl_pct=round(pnl_pct, 3),
            pnl_usdt=round(pnl, 4),
            duration_mins=round(duration_mins, 1),
            conditions={
                **self.position.get("entry_diagnostics", {}),
                "best_pnl_pct": round(self.position.get("best_pnl_pct", 0.0), 3),
                "worst_pnl_pct": round(self.position.get("worst_pnl_pct", 0.0), 3),
                "held_secs": round(duration_mins * 60.0, 1),
            },
        )
        
        self.trades.append(trade)
        self.position = None


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
D = "\033[0m";  BOLD = "\033[1m"

def _c(v, fmt="+.2f", sfx="%"):
    return f"{'G' if v>0 else 'R' if v<0 else ''}{v:{fmt}}{sfx}".replace("G", G).replace("R", R) + D

def _bar(r, w=20):
    n = max(0, min(w, round(r * w)))
    return G + "█" * n + D + "░" * (w - n)

def _mdd(trades, init):
    cap = peak = init; mdd = 0.0
    for t in trades:
        cap += t.pnl_usdt; peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak * 100)
    return mdd

def print_report(trades: list[Trade], cfg, n_candles: int, n_ticks: int) -> None:
    if not trades:
        print("\n✗ No trades taken.")
        return

    wins  = [t for t in trades if t.pnl_pct > 0]
    loss  = [t for t in trades if t.pnl_pct <= 0]
    tu    = sum(t.pnl_usdt for t in trades)
    aw    = sum(t.pnl_pct for t in wins)  / len(wins)  if wins else 0
    al    = sum(t.pnl_pct for t in loss)  / len(loss)  if loss else 0
    wr    = len(wins) / len(trades) * 100
    pf    = sum(t.pnl_usdt for t in wins) / abs(sum(t.pnl_usdt for t in loss)) if loss else float("inf")
    
    # Handle cfg as either a config object or just initial_capital
    if hasattr(cfg, 'initial_capital'):
        initial_capital = cfg.initial_capital
        timeframe_minutes = getattr(cfg, 'timeframe_minutes', 5)
    else:
        initial_capital = cfg  # cfg is just the initial capital value
        timeframe_minutes = 5  # Default
    
    final = initial_capital + tu
    roi   = (final - initial_capital) / initial_capital * 100
    mdd   = _mdd(trades, initial_capital)
    adur  = sum(t.duration_mins for t in trades) / len(trades)
    span  = (f"{trades[0].entry_time:%Y-%m-%d} → "
             f"{trades[-1].exit_time or trades[-1].entry_time:%Y-%m-%d}")

    W = 96
    print(f"\n{BOLD}{'═'*W}{D}")
    print(f"{BOLD}  ORDERFLOW STRATEGY — BACKTEST RESULTS{D}   ({span})")
    print(f"  Ticks: {n_ticks:,}   Candles: {n_candles:,}   Trades: {len(trades)}   "
          f"Capital: ${initial_capital:,.0f}   TF: {timeframe_minutes}m")
    print(f"{BOLD}{'═'*W}{D}")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'  SUMMARY':─<{W}}{D}")
    for lbl, val in [
        ("Total Trades",     str(len(trades))),
        ("Winning Trades",   f"{len(wins)}  ({wr:.1f}%)  {_bar(wr/100)}"),
        ("Profit Factor",    f"{pf:.2f}"),
        ("Avg Win",          _c(aw)),   ("Avg Loss",    _c(al)),
        ("Best Trade",       _c(max(t.pnl_pct for t in trades))),
        ("Worst Trade",      _c(min(t.pnl_pct for t in trades))),
        ("Avg Duration",     f"{adur/60:.1f}h"),
        ("Total PnL (USDT)", _c(tu, "+,.2f", " USDT")),
        ("ROI",              _c(roi)),
        ("Max Drawdown",     _c(mdd)),
        ("Final Capital",    f"${final:,.2f}"),
    ]:
        print(f"  {lbl:28s}  {val}")

    # ── trade list ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'  TRADE LIST':─<{W}}{D}")
    print(f"  {'#':>3}  {'Open':17}  {'Close':17}  {'Entry Reason':25}  {'Exit Reason':30}  {'PnL':>8}  {'Dur':>7}")
    print("  " + "─" * 92)
    for t in trades:
        c2 = t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else f"{Y}OPEN{D}     "
        dh, dm = int(t.duration_mins // 60), int(t.duration_mins % 60)
        ds = f"{dh}h{dm:02d}m" if dh else f"{dm}m"
        col = G if t.pnl_pct > 0 else R
        print(f"  {t.trade_id:>3}  "
              f"{t.entry_time.strftime('%Y-%m-%d %H:%M'):17}  {c2:17}  "
              f"{t.entry_reason[:25]:25}  {t.exit_reason[:30]:30}  "
              f"{col}{t.pnl_pct:>+6.2f}%{D}  {ds:>7}")

    # ── entry reasons ─────────────────────────────────────────────────────────
    def _table(title, groups):
        print(f"\n{BOLD}{title:─<{W}}{D}")
        print(f"  {'Reason':36}  {'#':>4}  {'Win%':>6}  {'Avg PnL':>8}  {'Total PnL':>14}")
        print("  " + "─" * 74)
        for reason, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
            w   = sum(1 for t in grp if t.pnl_pct > 0)
            wr_ = w / len(grp) * 100
            ap  = sum(t.pnl_pct  for t in grp) / len(grp)
            tp  = sum(t.pnl_usdt for t in grp)
            print(f"  {reason[:36]:36}  {len(grp):>4}  {wr_:>5.1f}%  "
                  f"{_c(ap):>18}  {_c(tp, '+,.2f', ' USDT'):>26}")

    eg: dict[str, list] = defaultdict(list)
    xg: dict[str, list] = defaultdict(list)
    for t in trades: eg[t.entry_reason].append(t); xg[t.exit_reason].append(t)
    _table("  ENTRY REASONS", eg)
    _table("  EXIT REASONS",  xg)




# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def export_csv(trades: list[Trade], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_id","entry_time","exit_time","entry_price","exit_price",
                    "entry_reason","exit_reason","pnl_pct","pnl_usdt","duration_mins"])
        for t in trades:
            w.writerow([t.trade_id,
                        t.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        t.exit_time.strftime("%Y-%m-%d %H:%M:%S") if t.exit_time else "",
                        round(t.entry_price,4), round(t.exit_price,4),
                        t.entry_reason, t.exit_reason,
                        t.pnl_pct, t.pnl_usdt, t.duration_mins])
    print(f"  ✓ Saved → {path}")


def _run_backtest_subprocess(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def run_parallel_backtests(commands: list[list[str]], workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_backtest_subprocess, cmd): cmd for cmd in commands}
        for fut in as_completed(futures):
            rc = fut.result()
            if rc != 0:
                raise RuntimeError(f"Parallel backtest failed with exit code {rc}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Live Strategy Backtester — uses actual live strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tick Parquet: columns [ts, price, qty, side]

Examples:
  python backtest.py --config nautilus/config/profiles/live.yaml --ticks ticks/BTCUSDT/2024-01.parquet
  python backtest.py --demo --config nautilus/config/profiles/live.yaml
  python backtest.py --config nautilus/config/profiles/live.yaml --ticks ticks/BTCUSDT/ --start 2024-01-01
        """
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to live strategy config (YAML)")
    p.add_argument("--ticks", type=str, 
                   help="Tick Parquet file or directory")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--start-time", type=str, help="Start time (HH:MM, e.g. 09:30)")
    p.add_argument("--end-time", type=str, help="End time (HH:MM, e.g. 16:00)")
    p.add_argument("--export", type=str, default="backtest_results.csv")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Speed mode: coarser flat throttle + evaluate entries once per new LTF candle",
    )
    p.add_argument(
        "--compress-ms",
        type=int,
        default=250,
        help="Compress ticks into VWAP buckets (e.g. 100, 250, 500). 0 disables.",
    )
    p.add_argument("--batch-size", type=int, default=200_000, help="Parquet streaming batch size.")
    p.add_argument("--heatmap-stride", type=int, default=8, help="Update heatmap every N ticks in fast mode.")
    p.add_argument("--vp-stride", type=int, default=4, help="Update volume profile every N ticks in fast mode.")
    p.add_argument(
        "--ranges",
        type=str,
        help="Comma-separated ranges for parallel runs: start:end,start:end",
    )
    p.add_argument("--workers", type=int, default=2, help="Parallel workers for --ranges.")
    args = p.parse_args()

    if args.ranges:
        if not args.ticks:
            print("Error: --ranges requires --ticks."); sys.exit(1)
        commands: list[list[str]] = []
        for item in args.ranges.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                st, en = item.split(":", 1)
            except ValueError:
                print(f"Invalid range entry: {item}"); sys.exit(1)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config", args.config,
                "--ticks", args.ticks,
                "--start", st,
                "--end", en,
                "--export", args.export,
                "--capital", str(args.capital),
                "--compress-ms", str(args.compress_ms),
                "--batch-size", str(args.batch_size),
                "--heatmap-stride", str(args.heatmap_stride),
                "--vp-stride", str(args.vp_stride),
            ]
            if args.fast:
                cmd.append("--fast")
            commands.append(cmd)
        if not commands:
            print("Error: --ranges provided but no valid entries found."); sys.exit(1)
        print(f"Running {len(commands)} backtests in parallel with {args.workers} workers...")
        run_parallel_backtests(commands, workers=max(1, args.workers))
        return

    # Initialize the live strategy backtester
    backtester = LiveStrategyBacktester(args.config, args.capital)
    tf = backtester.strategy_config.timeframe
    if tf.endswith("m"):
        tf_minutes = int(tf[:-1])
    elif tf.endswith("h"):
        tf_minutes = int(tf[:-1]) * 60
    else:
        tf_minutes = 5
    
    if args.demo:
        print("  Generating synthetic tick data…")
        ticks = generate_demo_ticks(n_candles=2000, tf_minutes=tf_minutes)
        ticks = filter_ticks_by_range(ticks, args.start, args.end, args.start_time, args.end_time)
        if args.compress_ms > 0:
            orig = len(ticks)
            ticks = compress_ticks(ticks, args.compress_ms)
            print(f"   Compression: {orig:,} -> {len(ticks):,} ticks (bucket={args.compress_ms}ms)")
        t0 = perf_counter()
        trades, n_candles = backtester.run_backtest([ticks], total_ticks=len(ticks), fast_mode=args.fast)
        runtime = perf_counter() - t0
        tps = len(ticks) / runtime if runtime > 0 else 0.0
        print(f"\n   Benchmark: {len(ticks):,} ticks in {runtime:.2f}s ({tps:,.0f} ticks/s)")
        print_report(trades, backtester.initial_capital, n_candles=n_candles, n_ticks=len(ticks))
        if trades:
            export_csv(trades, args.export)
        return
    
    elif args.ticks:
        path = Path(args.ticks)
        if not path.exists():
            print(f"Error: {path} not found."); sys.exit(1)

        sd_ms, ed_ms = _range_ms(args.start, args.end, args.start_time, args.end_time)
        total_ticks = estimate_parquet_ticks(str(path), sd_ms, ed_ms)
        if total_ticks <= 0:
            print("✗ No ticks loaded."); sys.exit(1)

        print(f"   ~{total_ticks:,} ticks estimated   Running live strategy stack…\n")
        tick_batches = stream_parquet_tick_batches(
            str(path),
            start=args.start,
            end=args.end,
            start_time=args.start_time,
            end_time=args.end_time,
            batch_size=args.batch_size,
            compress_ms=args.compress_ms,
        )
        t0 = perf_counter()
        trades, n_candles = backtester.run_backtest(
            tick_batches,
            total_ticks=total_ticks,
            fast_mode=args.fast,
            heatmap_stride=args.heatmap_stride,
            vp_stride=args.vp_stride,
        )
        runtime = perf_counter() - t0
        tps = total_ticks / runtime if runtime > 0 else 0.0
        print(f"\n   Benchmark: {total_ticks:,} ticks in {runtime:.2f}s ({tps:,.0f} ticks/s)")
        print_report(trades, backtester.initial_capital, n_candles=n_candles, n_ticks=total_ticks)
        if trades:
            export_csv(trades, args.export)
        return
    else:
        p.print_help(); sys.exit(0)


if __name__ == "__main__":
    main()
