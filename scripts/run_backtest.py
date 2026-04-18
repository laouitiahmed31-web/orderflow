"""
Fast tick-level Nautilus backtest runner for the existing OrderflowStrategy.

Loads tick Parquet files, wires the strategy into BacktestEngine, and exports:
- trade logs
- pnl summary
- equity curve
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, BookType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Ensure local project modules are importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautilus.config.loader import load_orderflow_config
from nautilus.config.schema import orderflow_strategy_config_from_stack
from nautilus.data.ticks import parquet_ticks_to_trade_ticks
from nautilus.strategy.orderflow_strategy import OrderflowStrategy


def _parse_iso_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _collect_parquet_files(parquet_path: Path) -> list[Path]:
    if parquet_path.is_file():
        if parquet_path.suffix.lower() != ".parquet":
            raise SystemExit(
                f"Expected a Parquet file (*.parquet), got: {parquet_path}. "
                "This runner only supports Parquet tick input."
            )
        return [parquet_path]
    files = sorted(parquet_path.glob("**/*.parquet"))
    if not files:
        raise SystemExit(
            f"No Parquet files found under: {parquet_path}. "
            "Provide a .parquet file or a directory containing .parquet files."
        )
    return [p for p in files if p.is_file()]


def build_engine(
    *,
    initial_balance: float,
    slippage_prob: float,
    latency_ms: int,
) -> tuple[BacktestEngine, object]:
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
            run_analysis=True,
        )
    )

    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        book_type=BookType.L1_MBP,
        base_currency=None,
        starting_balances=[Money(initial_balance, USDT)],
        fill_model=FillModel(
            prob_fill_on_limit=0.8,
            prob_slippage=slippage_prob,
            random_seed=42,
        ),
        latency_model=LatencyModel(
            base_latency_nanos=int(latency_ms * 1_000_000),
        ),
    )

    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    engine.add_instrument(instrument)
    return engine, instrument


def _load_tick_data(
    *,
    parquet_path: Path,
    instrument,
    start: datetime,
    end: datetime,
) -> list:
    ticks = []
    start_ms = int(start.timestamp() * 1e3)
    end_ms = int(end.timestamp() * 1e3)
    for file_path in _collect_parquet_files(parquet_path):
        chunk = parquet_ticks_to_trade_ticks(
            file_path,
            instrument,
            start_ts_ms=start_ms,
            end_ts_ms=end_ms,
        )
        if not chunk:
            continue
        ticks.extend(chunk)
    ticks.sort(key=lambda t: t.ts_event)
    return ticks


def _export_outputs(
    *,
    output_dir: Path,
    initial_balance: float,
    final_balance: float,
    metrics_dir: Path,
    fee_maker_bps: float,
    fee_taker_bps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pnl = final_balance - initial_balance
    pnl_pct = (pnl / initial_balance * 100.0) if initial_balance else 0.0

    metric_files = sorted(metrics_dir.glob("orderflow_metrics_*.jsonl"))
    if not metric_files:
        summary = {
            "initial_balance_usdt": initial_balance,
            "final_balance_usdt": final_balance,
            "net_pnl_usdt": pnl,
            "net_pnl_pct": pnl_pct,
            "estimated_fees_usdt": 0.0,
            "fee_adjusted_pnl_usdt": pnl,
            "fee_adjusted_pnl_pct": pnl_pct,
            "fee_assumptions_bps": {"maker": fee_maker_bps, "taker": fee_taker_bps},
        }
        (output_dir / "pnl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(columns=["event_idx", "event", "realized_pnl", "equity"]).to_csv(
            output_dir / "equity_curve.csv",
            index=False,
        )
        pd.DataFrame(columns=["event_idx", "event", "price", "qty", "side", "fee"]).to_csv(
            output_dir / "trade_logs.csv",
            index=False,
        )
        return

    rows: list[dict] = []
    for line in metric_files[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    fills = []
    equity_rows = []
    equity = initial_balance
    estimated_fees = 0.0
    event_idx = 0
    taker_fee_rate = fee_taker_bps / 10_000.0
    for row in rows:
        event_idx += 1
        event = row.get("event")
        data = row.get("data", {})
        if event == "fill":
            px = data.get("price")
            qty = data.get("qty")
            fee = data.get("fee")
            est_fee = 0.0
            if fee is None and px is not None and qty is not None:
                try:
                    est_fee = float(px) * float(qty) * taker_fee_rate
                except (TypeError, ValueError):
                    est_fee = 0.0
            elif fee is not None:
                try:
                    est_fee = float(fee)
                except (TypeError, ValueError):
                    est_fee = 0.0
            estimated_fees += est_fee
            fills.append(
                {
                    "event_idx": event_idx,
                    "event": event,
                    "price": px,
                    "qty": qty,
                    "side": data.get("side"),
                    "fee": fee,
                    "estimated_fee": est_fee,
                }
            )
        elif event == "position_closed":
            realized = float(data.get("realized_pnl", 0.0))
            equity += realized
            equity_rows.append(
                {
                    "event_idx": event_idx,
                    "event": event,
                    "realized_pnl": realized,
                    "equity": equity,
                }
            )

    pd.DataFrame(fills).to_csv(output_dir / "trade_logs.csv", index=False)
    pd.DataFrame(equity_rows).to_csv(output_dir / "equity_curve.csv", index=False)
    fee_adj_pnl = pnl - estimated_fees
    fee_adj_pct = (fee_adj_pnl / initial_balance * 100.0) if initial_balance else 0.0
    summary = {
        "initial_balance_usdt": initial_balance,
        "final_balance_usdt": final_balance,
        "net_pnl_usdt": pnl,
        "net_pnl_pct": pnl_pct,
        "estimated_fees_usdt": estimated_fees,
        "fee_adjusted_pnl_usdt": fee_adj_pnl,
        "fee_adjusted_pnl_pct": fee_adj_pct,
        "fee_assumptions_bps": {"maker": fee_maker_bps, "taker": fee_taker_bps},
    }
    (output_dir / "pnl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    metrics_dir = output_dir / "metrics"
    print(f"Backtest range: {args.start.isoformat()} -> {args.end.isoformat()}")
    print(f"Parquet source: {args.parquet}")
    print(f"Output dir:     {output_dir}")

    engine, instrument = build_engine(
        initial_balance=args.balance,
        slippage_prob=args.slippage_prob,
        latency_ms=args.latency_ms,
    )

    print("Loading strategy config ...")
    stack_cfg = load_orderflow_config(args.config)
    base_cfg = orderflow_strategy_config_from_stack(stack_cfg)
    cfg_keys = getattr(base_cfg.__class__, "__annotations__", {}).keys()
    cfg_values = {k: getattr(base_cfg, k) for k in cfg_keys}
    cfg_values.update(
        {
            "instrument_id": instrument.id,
            "require_orderbook": False,
            "log_metrics": True,
            "metrics_dir": str(metrics_dir),
        }
    )
    strategy_cfg = base_cfg.__class__(**cfg_values)
    strategy = OrderflowStrategy(config=strategy_cfg)
    # Backtest must use engine portfolio/fills, not the strategy's paper wallet.
    if hasattr(strategy, "_paper_mode"):
        strategy._paper_mode = False
    engine.add_strategy(strategy)

    print("Loading tick Parquet ...")
    ticks = _load_tick_data(
        parquet_path=args.parquet,
        instrument=instrument,
        start=args.start,
        end=args.end,
    )
    if not ticks:
        raise SystemExit("No ticks loaded for selected range.")
    print(f"Ticks loaded: {len(ticks):,}")
    engine.add_data(ticks)

    print("Running backtest ...")
    engine.run(start=args.start, end=args.end)

    account = engine.portfolio.account(Venue("BINANCE"))
    final_balance = args.balance
    if account:
        bal = account.balance(USDT)
        if bal:
            final_balance = float(bal.total.as_double())

    _export_outputs(
        output_dir=output_dir,
        initial_balance=args.balance,
        final_balance=final_balance,
        metrics_dir=metrics_dir,
        fee_maker_bps=args.fee_maker_bps,
        fee_taker_bps=args.fee_taker_bps,
    )
    print(f"Final balance: {final_balance:,.2f} USDT")
    print(f"PnL summary:   {(output_dir / 'pnl_summary.json')}")
    print(f"Trade logs:    {(output_dir / 'trade_logs.csv')}")
    print(f"Equity curve:  {(output_dir / 'equity_curve.csv')}")
    engine.dispose()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast tick-level Nautilus backtest runner.")
    p.add_argument("--config", type=Path, required=True, help="Strategy YAML (existing strategy config).")
    p.add_argument("--parquet", type=Path, required=True, help="Parquet file or directory of files.")
    p.add_argument("--start", required=True, type=_parse_iso_utc, help="ISO datetime, UTC if naive.")
    p.add_argument("--end", required=True, type=_parse_iso_utc, help="ISO datetime, UTC if naive.")
    p.add_argument("--balance", type=float, default=10_000.0, help="Initial USDT balance.")
    p.add_argument("--fee-maker-bps", type=float, default=2.0, help="Maker fee in bps.")
    p.add_argument("--fee-taker-bps", type=float, default=4.0, help="Taker fee in bps.")
    p.add_argument("--slippage-prob", type=float, default=0.30, help="Fill slippage probability [0,1].")
    p.add_argument("--latency-ms", type=int, default=50, help="Base latency in milliseconds.")
    p.add_argument("--output-dir", type=Path, default=Path("backtest_output"), help="Output folder.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
