#!/usr/bin/env python3
"""
backtest.py — Standalone OrderFlow Backtester
==============================================
Plugs directly into orderflow_indicators.py (your actual indicator code).
No external framework needed.

DATA PIPELINE:
  raw tick CSV  →  ticks_to_candle_flow()  →  populate_flows()  →  strategy  →  tables

TICK CSV FORMAT (one row per trade):
  ts_ms, price, qty, side
  1704067200000, 43521.50, 0.012, buy
  1704067200120, 43519.80, 0.045, sell
  ...

Also accepts Binance aggTrades CSV format automatically.

Usage:
  python backtest.py --ticks data/btcusdt_ticks.csv
  python backtest.py --ticks data/ticks/ --timeframe 5
  python backtest.py --demo
  python backtest.py --demo --stoploss 0.03 --stack-min 2
  python backtest.py --ticks data/ --timeframe 15 --export results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Import your actual indicator code ─────────────────────────────────────────
try:
    from orderflow_indicators import (
        CandleFlow,
        ticks_to_candle_flow,
        populate_flows,
    )
    _INDICATORS_AVAILABLE = True
except ImportError:
    _INDICATORS_AVAILABLE = False
    print("⚠  orderflow_indicators.py not found — using built-in fallback.")
    print("   Place backtest.py in the same directory as orderflow_indicators.py.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class BacktestConfig:
    # Strategy thresholds (mirrors main.py defaults)
    imbalance_threshold:  float = 0.25
    cvd_smoothing:        int   = 5
    absorption_min:       float = 0.3   # Sell absorption exit threshold (absorption in [-1,+1])
    stack_min_rows:       int   = 3
    ob_imb_threshold:     float = 0.0   # 0.0 = disabled in backtest (no live OB available from ticks)
    large_vol_ratio_min:  float = 1.3   # Large buy dominance: 30%+ of total large volume
    # Risk
    position_size:        float = 0.95
    stoploss:             float = -0.02
    trailing_offset:      float = 0.01
    trailing_trigger:     float = 0.015
    fee_rate:             float = 0.001
    slippage:             float = 0.0005
    initial_capital:      float = 10_000.0
    # Data pipeline
    timeframe_minutes:    int   = 5
    large_trade_pct:      float = 0.90
    price_bucket_size:    float = 1.0
    divergence_window:    int   = 3


# ══════════════════════════════════════════════════════════════════════════════
# EMA (same as main.py)
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, period: int):
        self.alpha = 2 / (period + 1)
        self.value: Optional[float] = None

    def update(self, v: float) -> float:
        if self.value is None:
            self.value = v
        else:
            self.value = self.alpha * v + (1 - self.alpha) * self.value
        return self.value


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
def load_ticks_csv(path: str) -> list[Tick]:
    """
    Two formats accepted:

    A) Simple:  ts_ms, price, qty, side
    B) Binance aggTrades: agg_id, price, qty, first_id, last_id, ts_ms, is_buyer_maker
       is_buyer_maker=True → aggressive SELL (buyer is maker = seller is taker)
    """
    ticks = []
    count = 0
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
                ticks.append(Tick(ts=ts_ms, price=price, qty=qty, side=side))
                count += 1
                if count % 500_000 == 0:
                    print(f"       Loaded {count:,} ticks…", flush=True)
            except (ValueError, IndexError):
                continue
    print(f"      ✓ {count:,} total ticks loaded", flush=True)
    return sorted(ticks, key=lambda t: t.ts)


def load_ticks_dir(directory: str, start: str = None, end: str = None) -> list[Tick]:
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
        all_ticks.extend(load_ticks_csv(str(f)))
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


def _build_candle(tl, open_ts, tf_ms, cfg, flows_list, closes_list):
    """Build single candle from tick list, update running lists."""
    if not tl:
        return None
    bv = sum(t["qty"] for t in tl if t["side"].upper() == "BUY")
    sv = sum(t["qty"] for t in tl if t["side"].upper() == "SELL")
    tv = bv + sv
    delta = bv - sv
    qtys = sorted(t["qty"] for t in tl)
    thresh = qtys[int(len(qtys) * cfg.large_trade_pct)] if qtys else 0
    lb = sum(t["qty"] for t in tl if t["side"].upper() == "BUY" and t["qty"] > thresh)
    ls = sum(t["qty"] for t in tl if t["side"].upper() == "SELL" and t["qty"] > thresh)
    prices = [t["price"] for t in tl]
    cp = tl[-1]["price"]
    
    # Build CandleFlow
    f = CandleFlow(
        open_ts=open_ts, close_ts=open_ts+tf_ms,
        buy_vol=bv, sell_vol=sv, delta=delta, total_vol=tv,
        large_buy_vol=lb, large_sell_vol=ls,
        max_price=max(prices) if prices else cp,
        min_price=min(prices) if prices else cp,
        close_price=cp
    ) if _INDICATORS_AVAILABLE else type('F', (), {
        'open_ts': open_ts, 'close_ts': open_ts+tf_ms,
        'buy_vol': bv, 'sell_vol': sv, 'delta': delta, 'total_vol': tv,
        'large_buy_vol': lb, 'large_sell_vol': ls,
        'max_price': max(prices) if prices else cp,
        'min_price': min(prices) if prices else cp,
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
# BACKTESTER ENGINE (exact logic from main.py)
# ══════════════════════════════════════════════════════════════════════════════
class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.cvd_ema = EMA(cfg.cvd_smoothing)
        self.prev_cvd_ema: Optional[float] = None
        self.in_position = False
        self.entry_price = 0.0
        self.trailing_active = False
        self.trailing_peak = 0.0
        self.capital = cfg.initial_capital
        self.trades: list[Trade] = []
        self._counter = 0
        self._open: Optional[Trade] = None

    def run(self, flows: list) -> list[Trade]:
        for f in flows:
            self._on_bar(f)
        if self.in_position and self._open and flows:
            last = flows[-1]
            self._exit("end_of_data", last.close_price,
                       datetime.utcfromtimestamp(last.close_ts / 1000))
        return self.trades

    def _ts(self, f) -> datetime:
        return datetime.utcfromtimestamp(f.open_ts / 1000)

    def _on_bar(self, f):
        cvd_now = self.cvd_ema.update(f.cvd)
        rising  = self.prev_cvd_ema is not None and cvd_now > self.prev_cvd_ema
        self.prev_cvd_ema = cvd_now
        if self.in_position:
            self._check_exit(f, rising)
            return
        self._check_entry(f, rising)

    # ── entry ──────────────────────────────────────────────────────────────────
    def _check_entry(self, f, cvd_rising: bool):
        ls = f.large_buy_vol + f.large_sell_vol
        dom = (f.large_buy_vol - f.large_sell_vol) / ls if ls > 1e-9 else 0.0
        conds = {
            "cvd_rising":    cvd_rising,
            "imbalance":     f.imbalance    >= self.cfg.imbalance_threshold,
            "no_absorption": f.absorption   >= 0,
            "stacked_imb":   f.stacked_imb  >= self.cfg.stack_min_rows,
            "ob_imbalance":  f.ob_imbalance >= self.cfg.ob_imb_threshold,
            "large_dom":     dom            >= (self.cfg.large_vol_ratio_min - 1),
            "no_delta_div":  f.delta_div    != 1.0,  # Only block bearish divergence (-1.0), allow bullish
        }
        if all(conds.values()):
            self._enter(f, conds)

    def _enter(self, f, conds: dict):
        fill = f.close_price * (1 + self.cfg.slippage)
        used = self.capital * self.cfg.position_size
        if used < 11: return
        qty = (used - used * self.cfg.fee_rate) / fill
        self.capital -= used
        self.in_position = True
        self.entry_price = fill
        self.trailing_active = False
        self.trailing_peak = fill
        self._counter += 1
        self._open = Trade(
            trade_id=self._counter, entry_time=self._ts(f),
            entry_price=fill, qty=qty, capital_used=used,
            entry_reason=self._label(conds, f), conditions=conds)

    def _label(self, conds: dict, f) -> str:
        parts = ["cvd_up"] if conds["cvd_rising"] else []
        parts.append(f"stack_{int(f.stacked_imb)}" if f.stacked_imb >= 5 else "stack_3+")
        if f.absorption > 0.10: parts.append("buy_abs")
        parts.append("ob_strong" if f.ob_imbalance >= 0.30 else "ob_bid")
        parts.append("strong_imb" if f.imbalance >= 0.50 else "imb_ok")
        return "+".join(parts)

    # ── exit ───────────────────────────────────────────────────────────────────
    def _check_exit(self, f, cvd_rising: bool):
        price   = f.close_price
        pnl_pct = (price - self.entry_price) / self.entry_price

        if pnl_pct <= self.cfg.stoploss:
            self._exit("stoploss", price, self._ts(f)); return

        if pnl_pct >= self.cfg.trailing_trigger:
            self.trailing_active = True
        if self.trailing_active:
            self.trailing_peak = max(self.trailing_peak, price)
            if (price - self.trailing_peak) / self.trailing_peak <= -self.cfg.trailing_offset:
                self._exit("trailing_stop", price, self._ts(f)); return

        reasons = []
        if not cvd_rising:                              reasons.append("cvd_rolling_over")
        if f.absorption <= -self.cfg.absorption_min:   reasons.append("sell_absorption")  # ⚠ DEAD
        if f.delta_div == 1.0:                          reasons.append("bearish_delta_div")
        if f.imbalance <= -self.cfg.imbalance_threshold: reasons.append("imbalance_flipped")

        if reasons:
            self._exit("+".join(reasons), price, self._ts(f))

    def _exit(self, reason: str, price: float, time: datetime):
        if not self.in_position or not self._open: return
        fill = price * (1 - self.cfg.slippage)
        fee  = fill * self._open.qty * self.cfg.fee_rate
        net  = (fill - self.entry_price) * self._open.qty - fee
        self.capital += self._open.capital_used + net
        dur = (time - self._open.entry_time).total_seconds() / 60
        self._open.exit_time = time; self._open.exit_price = fill
        self._open.exit_reason = reason
        self._open.pnl_pct = round(net / self._open.capital_used * 100, 3)
        self._open.pnl_usdt = round(net, 4)
        self._open.duration_mins = round(dur, 1)
        self.trades.append(self._open)
        self._open = None; self.in_position = False; self.trailing_active = False


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

def print_report(trades: list[Trade], cfg: BacktestConfig,
                 n_candles=0, n_ticks=0):
    W = 96

    if not trades:
        print(f"\n{Y}⚠  No trades generated.{D}")
        print(f"   Ticks: {n_ticks:,}   Candles: {n_candles:,}")
        print(f"   Try: --stack-min 2  --imbalance 0.15  --ob-thresh 0.05")
        return

    wins  = [t for t in trades if t.pnl_pct > 0]
    loss  = [t for t in trades if t.pnl_pct <= 0]
    tu    = sum(t.pnl_usdt for t in trades)
    aw    = sum(t.pnl_pct for t in wins)  / len(wins)  if wins else 0
    al    = sum(t.pnl_pct for t in loss)  / len(loss)  if loss else 0
    wr    = len(wins) / len(trades) * 100
    pf    = sum(t.pnl_usdt for t in wins) / abs(sum(t.pnl_usdt for t in loss)) if loss else float("inf")
    final = cfg.initial_capital + tu
    roi   = (final - cfg.initial_capital) / cfg.initial_capital * 100
    mdd   = _mdd(trades, cfg.initial_capital)
    adur  = sum(t.duration_mins for t in trades) / len(trades)
    span  = (f"{trades[0].entry_time:%Y-%m-%d} → "
             f"{trades[-1].exit_time or trades[-1].entry_time:%Y-%m-%d}")

    print(f"\n{BOLD}{'═'*W}{D}")
    print(f"{BOLD}  ORDERFLOW STRATEGY — BACKTEST RESULTS{D}   ({span})")
    print(f"  Ticks: {n_ticks:,}   Candles: {n_candles:,}   Trades: {len(trades)}   "
          f"Capital: ${cfg.initial_capital:,.0f}   TF: {cfg.timeframe_minutes}m")
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="OrderFlow Backtester — raw ticks → indicators → tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tick CSV:  ts_ms,price,qty,side          (simple)
           or Binance aggTrades format   (auto-detected)

Examples:
  python backtest.py --demo
  python backtest.py --ticks btcusdt_ticks.csv --start 2024-01-15 --end 2024-01-20
  python backtest.py --ticks data/ticks/ --timeframe 5
  python backtest.py --demo --stoploss 0.03 --stack-min 2 --imbalance 0.20
  python backtest.py --ticks data/ --start 2024-01-01 --start-time 09:30 --end-time 16:00
        """)
    p.add_argument("--ticks",      type=str,   help="Tick CSV file or directory")
    p.add_argument("--demo",       action="store_true")
    p.add_argument("--start",      type=str,   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",        type=str,   help="End date (YYYY-MM-DD, inclusive)")
    p.add_argument("--start-time", type=str,   help="Start time (HH:MM, e.g. 09:30)")
    p.add_argument("--end-time",   type=str,   help="End time (HH:MM, e.g. 16:00)")
    p.add_argument("--export",     type=str,   default="backtest_results.csv")
    p.add_argument("--timeframe",  type=int,   default=5,     help="Candle minutes (default 5)")
    p.add_argument("--imbalance",  type=float, default=0.25)
    p.add_argument("--stack-min",  type=int,   default=3)
    p.add_argument("--ob-thresh",  type=float, default=0.0,
                   help="OB imbalance threshold (default 0 = disabled, no live OB in backtest)")
    p.add_argument("--stoploss",   type=float, default=0.02,  help="e.g. 0.02 = 2%%")
    p.add_argument("--capital",    type=float, default=10_000.0)
    args = p.parse_args()

    cfg = BacktestConfig(
        imbalance_threshold = args.imbalance,
        stack_min_rows      = args.stack_min,
        ob_imb_threshold    = args.ob_thresh,
        stoploss            = -abs(args.stoploss),
        initial_capital     = args.capital,
        timeframe_minutes   = args.timeframe,
    )

    if args.demo:
        print("  Generating synthetic tick data…")
        ticks = generate_demo_ticks(n_candles=2000, tf_minutes=cfg.timeframe_minutes)
        flows = ticks_to_candles(ticks, cfg)
        trades = Backtester(cfg).run(flows)
        print_report(trades, cfg, n_candles=len(flows), n_ticks=len(ticks))
        if trades:
            export_csv(trades, args.export)
        return
    
    elif args.ticks:
        path = Path(args.ticks)
        if path.is_dir():
            print(f"  Loading ticks from {path}/ (streaming)…")
            csv_files = sorted(path.glob("**/*.csv"))
            if not csv_files:
                print(f"Error: No CSV files in {path}"); sys.exit(1)
            
            import re
            all_flows = []
            all_closes = []
            sd = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
            ed = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
            
            for csv_f in csv_files:
                fname = csv_f.name
                fdt_start = None
                fdt_end = None
                
                # Try monthly format (YYYY-MM)
                m = re.search(r'(\d{4})-(\d{2})', fname)
                if m:
                    try:
                        year, month = int(m.group(1)), int(m.group(2))
                        fdt_start = datetime(year, month, 1)
                        # End of month
                        if month == 12:
                            fdt_end = datetime(year + 1, 1, 1) - __import__('datetime').timedelta(days=1)
                        else:
                            fdt_end = datetime(year, month + 1, 1) - __import__('datetime').timedelta(days=1)
                    except ValueError:
                        pass
                
                # Try daily format (YYYYMMDD) if not monthly
                if not fdt_start:
                    m = re.search(r'(\d{8})', fname)
                    if m:
                        try:
                            fdt_start = datetime.strptime(m.group(1), "%Y%m%d")
                            fdt_end = fdt_start
                        except ValueError:
                            pass
                
                # Check if file's date range overlaps with requested range
                if fdt_start and fdt_end:
                    # Skip if file ends before range starts or starts after range ends
                    if ed and fdt_start > ed.replace(hour=23, minute=59, second=59):
                        continue
                    if sd and fdt_end < sd.replace(hour=0, minute=0, second=0):
                        continue
                
                print(f"   Processing {fname}…")
                for f in ticks_to_candles_streaming(str(csv_f), cfg, args.start, args.end, args.start_time, args.end_time):
                    if f:
                        all_flows.append(f)
                        all_closes.append(f.close_price)
            
            flows = all_flows
        elif path.is_file():
            print(f"  Loading {path} (streaming)…")
            all_flows = []
            all_closes = []
            for f in ticks_to_candles_streaming(str(path), cfg, args.start, args.end, args.start_time, args.end_time):
                if f:
                    all_flows.append(f)
                    all_closes.append(f.close_price)
            flows = all_flows
        else:
            print(f"Error: {path} not found."); sys.exit(1)
    else:
        p.print_help(); sys.exit(0)

    if not flows:
        print("✗ No candles generated."); sys.exit(1)

    # Populate derived metrics (CVD, imbalance, absorption, etc.)
    if _INDICATORS_AVAILABLE:
        populate_flows(
            flows=flows,
            closes=[f.close_price for f in flows],
            running_cvd=0.0,
            ob_imbalance=0.0,
            divergence_window=cfg.divergence_window,
        )

    print(f"   {len(flows):,} candles   Running strategy…\n")

    trades = Backtester(cfg).run(flows)
    print_report(trades, cfg, n_candles=len(flows), n_ticks=0)
    if trades:
        export_csv(trades, args.export)


if __name__ == "__main__":
    main()
