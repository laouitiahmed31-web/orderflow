"""
Load ``OrderflowNautilusConfig`` from YAML (optional PyYAML) or JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nautilus_trader.model.identifiers import InstrumentId

from nautilus.config.schema import ExecutionParams
from nautilus.config.schema import OrderflowNautilusConfig
from nautilus.config.schema import RiskParams
from nautilus.config.schema import SignalParams
from nautilus.config.schema import SignalsConfig


def _load_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(text)
        except ImportError as e:
            raise ImportError(
                "Install PyYAML to use YAML configs: pip install pyyaml",
            ) from e
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_orderflow_config(path: str | Path) -> OrderflowNautilusConfig:
    raw = _load_file(Path(path))
    profile = raw.get("profile", "live")
    inst = InstrumentId.from_str(raw["instrument_id"])
    signal_raw = raw.get("signal", {}) or {}
    risk_raw = raw.get("risk", {}) or {}
    exec_raw = raw.get("execution", {}) or {}
    signals_raw = raw.get("signals_config", raw.get("signals", {})) or {}

    signal = SignalParams(
        imbalance_threshold=float(signal_raw.get("imbalance_threshold", 0.25)),
        cvd_smoothing=int(signal_raw.get("cvd_smoothing", 5)),
        # Directional absorption is in [-1, +1].
        # Entry gate uses: absorption >= -absorption_min
        absorption_min=float(signal_raw.get("absorption_min", 0.15)),
        stack_min_rows=int(signal_raw.get("stack_min_rows", 3)),
        ob_imb_threshold=float(signal_raw.get("ob_imb_threshold", 0.15)),
        # Large-trade dominance ratio threshold in [-1, +1].
        large_vol_ratio_min=float(signal_raw.get("large_vol_ratio_min", 0.10)),
        price_bucket_size=float(signal_raw.get("price_bucket_size", 1.0)),
        large_trade_pct=float(signal_raw.get("large_trade_pct", 0.90)),
        divergence_window=int(signal_raw.get("divergence_window", 3)),
    )
    risk = RiskParams(
        # Keep defaults conservative and aligned with schema.py.
        max_position_fraction=float(risk_raw.get("max_position_fraction", 0.10)),
        max_notional_usdt=risk_raw.get("max_notional_usdt"),
        max_leverage=risk_raw.get("max_leverage"),
        max_daily_loss_pct=float(risk_raw.get("max_daily_loss_pct", 5.0)),
        max_consecutive_losses=int(risk_raw.get("max_consecutive_losses", 5)),
        max_spread_bps=float(risk_raw.get("max_spread_bps", 25.0)),
        stale_tick_ms=float(risk_raw.get("stale_tick_ms", 5000.0)),
        min_top_of_book_qty=float(risk_raw.get("min_top_of_book_qty", 0.0)),
        kill_switch_path=risk_raw.get("kill_switch_path"),
        force_exit_path=risk_raw.get("force_exit_path", "orderflow/.force_exit"),
        loss_cooldown_secs=float(risk_raw.get("loss_cooldown_secs", 0.0)),
        min_hold_secs=float(risk_raw.get("min_hold_secs", 0.0)),
        risk_per_trade_pct=float(risk_raw.get("risk_per_trade_pct", 0.0025)),
        min_stop_bps=float(risk_raw.get("min_stop_bps", 12.0)),
        max_stop_bps=float(risk_raw.get("max_stop_bps", 250.0)),
    )
    execution = ExecutionParams(
        # Prefer safer, more configurable execution defaults.
        use_market_entries=bool(exec_raw.get("use_market_entries", False)),
        entry_post_only=bool(exec_raw.get("entry_post_only", True)),
        stoploss_pct=float(exec_raw.get("stoploss_pct", 0.02)),
        trailing_trigger_pct=float(exec_raw.get("trailing_trigger_pct", 0.015)),
        trailing_offset_pct=float(exec_raw.get("trailing_offset_pct", 0.01)),
        max_time_in_trade_secs=exec_raw.get("max_time_in_trade_secs"),
        max_entry_drift_bps=float(exec_raw.get("max_entry_drift_bps", 8.0)),
        backtest_fee_rate=float(exec_raw.get("backtest_fee_rate", 0.0004)),
        backtest_slippage_rate=float(exec_raw.get("backtest_slippage_rate", 0.0002)),
        min_structural_rr=float(exec_raw.get("min_structural_rr", 1.2)),
        max_structural_stop_bps=float(exec_raw.get("max_structural_stop_bps", 35.0)),
        short_rejection_min_confidence=float(exec_raw.get("short_rejection_min_confidence", 1.05)),
        short_rejection_size_mult=float(exec_raw.get("short_rejection_size_mult", 0.6)),
        early_invalidation_secs=float(exec_raw.get("early_invalidation_secs", 120.0)),
        early_invalidation_loss_pct=float(exec_raw.get("early_invalidation_loss_pct", 0.0018)),
        min_hold_secs=float(exec_raw.get("min_hold_secs", 10.0)),
        acceptance_failure_secs=float(exec_raw.get("acceptance_failure_secs", 60.0)),
        acceptance_failure_evals=int(exec_raw.get("acceptance_failure_evals", 2)),
    )
    signals_config = SignalsConfig(
        long=list(signals_raw.get("long", ["hvn_absorption_long"])),
        short=list(signals_raw.get("short", [])),
        require_all=bool(signals_raw.get("require_all", False)),
        module_kwargs=dict(signals_raw.get("module_kwargs", {})),
    )

    return OrderflowNautilusConfig(
        profile=profile,
        instrument_id=inst,
        order_id_tag=str(raw.get("order_id_tag", "OF-001")),
        client_id=str(raw.get("client_id", "BINANCE")),
        timeframe=str(raw.get("timeframe", "5m")),
        lookback_candles=int(raw.get("lookback_candles", 50)),
        book_depth=int(raw.get("book_depth", 5)),
        book_type=str(raw.get("book_type", "L2_MBP")),
        eval_throttle_ms=float(raw.get("eval_throttle_ms", 200.0)),
        require_orderbook=bool(raw.get("require_orderbook", True)),
        signals_config=signals_config,
        vp_config=raw.get("vp_config"),
        heatmap_config=raw.get("heatmap_config"),
        signal=signal,
        risk=risk,
        execution=execution,
        binance_environment=raw.get("binance_environment"),
    )
