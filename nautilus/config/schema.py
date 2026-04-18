"""
config/schema.py — Complete production config schema.

Added fields vs previous version:
  loss_cooldown_secs : float  — seconds to block new entries after a loss (default 60)
  min_hold_secs      : float  — minimum seconds to hold before signal exits fire (default 10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId


@dataclass
class SignalsConfig:
    """
    Parsed from YAML:
    signals:
      long:  [absorption_breakout_long, imbalance_continuation_long]
      short: [absorption_breakout_short]
      require_all: false   # fire on first passing module
      module_kwargs:       # passed to every module __init__
        imbalance_threshold: 0.25
        absorption_min: 0.15
    """
    long: list[str] = field(default_factory=lambda: ["hvn_absorption_long"])
    short: list[str] = field(default_factory=list)
    require_all: bool = False       # False = first module wins; True = all must pass
    module_kwargs: dict = field(default_factory=dict)


Profile = Literal["backtest", "paper", "live"]


class OrderflowStrategyConfig(StrategyConfig, frozen=True, kw_only=True):

    instrument_id: InstrumentId
    order_id_tag: str = "OF-001"
    client_id: str = "BINANCE"
    timeframe: str = "5m"
    htf_timeframe: str = "1h"
    lookback_candles: int = 50
    book_depth: int = 5
    book_type: str = "L2_MBP"
    eval_throttle_ms: float = 200.0
    require_orderbook: bool = False #set to True for live trading, can be False for backtesting without OB data

    # ── Structure ──────────────────────────────────────────────────────────
    swing_window: int = 5

    # ── Session filter ─────────────────────────────────────────────────────
    sessions_config: list | None = None

    # ── Signals ───────────────────────────────────────────────────────────
    signals_config: SignalsConfig | None = None
    vp_config: dict | None = None
    heatmap_config: dict | None = None

    # ── Indicator params ──────────────────────────────────────────────────
    imbalance_threshold: float = 0.25
    cvd_smoothing: int = 5
    absorption_min: float = 0.15
    stack_min_rows: int = 3
    ob_imb_threshold: float = 0.15
    large_vol_ratio_min: float = 0.10
    price_bucket_size: float = 1.0
    large_trade_pct: float = 0.90
    divergence_window: int = 3

    # ── Risk ──────────────────────────────────────────────────────────────
    max_position_fraction: float = 0.10
    max_notional_usdt: float | None = 500.0
    max_leverage: float | None = 2.0
    max_daily_loss_pct: float = 3.0
    max_consecutive_losses: int = 4
    max_spread_bps: float = 20.0
    stale_tick_ms: float = 5000.0
    min_top_of_book_qty: float = 0.0
    kill_switch_path: str | None = None
    # If this path exists while a position is open, the next eval issues a market close
    # (then deletes the file). Does not block entries (unlike kill_switch).
    force_exit_path: str | None = "orderflow/.force_exit"
    equity_state_path: str | None = "orderflow/.equity_state.json"
    # Risk-based sizing (institutional default): risk fixed fraction per trade
    risk_per_trade_pct: float = 0.0025          # 0.25% of equity per trade
    min_stop_bps: float = 12.0                  # reject too-tight stops (noise)
    max_stop_bps: float = 250.0                 # reject too-wide stops (tail risk)

    # ── Execution ─────────────────────────────────────────────────────────
    use_market_entries: bool = False
    entry_post_only: bool = True
    stoploss_pct: float = 0.018
    target_pct: float = 0.036
    trailing_trigger_pct: float = 0.012
    trailing_offset_pct: float = 0.008
    max_time_in_trade_secs: float | None = 3600.0
    max_entry_drift_bps: float = 8.0
    backtest_fee_rate: float = 0.0004
    backtest_slippage_rate: float = 0.0002
    min_structural_rr: float = 1.2
    max_structural_stop_bps: float = 35.0
    short_rejection_min_confidence: float = 1.05
    short_rejection_size_mult: float = 0.6
    early_invalidation_secs: float = 120.0
    early_invalidation_loss_pct: float = 0.0018
    acceptance_failure_evals: int = 2
    acceptance_failure_secs: float = 60.0

    # ── Trade pacing ───────────────────────────────────────────────────────────
    # FIX 9: These were both 0.0 (disabled) while the docstring claimed they were
    # enabled. 0 cooldown = immediate re-entry after a loss = death spiral risk.
    # Set to conservative non-zero defaults.
    loss_cooldown_secs: float = 60.0   # 1-minute block after any losing close
    min_hold_secs: float = 10.0        # hold at least 10 s before signal exits fire

    # ── ML ────────────────────────────────────────────────────────────────
    ml_state_path: str = "orderflow/.ml_state.pkl"

    # ── Ops ───────────────────────────────────────────────────────────────
    log_metrics: bool = True
    metrics_dir: str = "/home/adem/orderflow/orderflow/logs/metrics"


# ── Supporting config classes ───────────────────────────────────────────────

@dataclass
class SignalParams:
    imbalance_threshold: float = 0.25
    cvd_smoothing: int = 5
    absorption_min: float = 0.15
    stack_min_rows: int = 3
    ob_imb_threshold: float = 0.15
    large_vol_ratio_min: float = 0.10
    price_bucket_size: float = 1.0
    large_trade_pct: float = 0.90
    divergence_window: int = 3


@dataclass
class RiskParams:
    max_position_fraction: float = 0.10
    max_notional_usdt: float | None = None
    max_leverage: float | None = None
    max_daily_loss_pct: float = 5.0
    max_consecutive_losses: int = 5
    max_spread_bps: float = 25.0
    stale_tick_ms: float = 5000.0
    min_top_of_book_qty: float = 0.0
    kill_switch_path: str | None = None
    force_exit_path: str | None = "orderflow/.force_exit"
    loss_cooldown_secs: float = 0.0
    min_hold_secs: float = 0.0
    risk_per_trade_pct: float = 0.0025
    min_stop_bps: float = 12.0
    max_stop_bps: float = 250.0


@dataclass
class ExecutionParams:
    use_market_entries: bool = False
    entry_post_only: bool = True
    stoploss_pct: float = 0.02
    trailing_trigger_pct: float = 0.015
    trailing_offset_pct: float = 0.01
    max_time_in_trade_secs: int | None = None
    max_entry_drift_bps: float = 8.0
    backtest_fee_rate: float = 0.0004
    backtest_slippage_rate: float = 0.0002
    min_structural_rr: float = 1.2
    max_structural_stop_bps: float = 35.0
    short_rejection_min_confidence: float = 1.05
    short_rejection_size_mult: float = 0.6
    early_invalidation_secs: float = 120.0
    early_invalidation_loss_pct: float = 0.0018
    # NOTE: This is consumed by live strategy exit logic; keeping it under
    # execution matches YAML profiles which treat it as an execution constraint.
    min_hold_secs: float = 10.0
    acceptance_failure_secs: float = 60.0
    acceptance_failure_evals: int = 2


@dataclass
class OrderflowNautilusConfig:
    profile: str
    instrument_id: InstrumentId
    order_id_tag: str = "OF-001"
    client_id: str = "BINANCE"
    timeframe: str = "5m"
    lookback_candles: int = 50
    book_depth: int = 5
    book_type: str = "L2_MBP"
    eval_throttle_ms: float = 200.0
    require_orderbook: bool = True
    signals_config: SignalsConfig | None = None
    vp_config: dict | None = None
    heatmap_config: dict | None = None
    signal: SignalParams | None = None
    risk: RiskParams | None = None
    execution: ExecutionParams | None = None
    binance_environment: str | None = None


def orderflow_strategy_config_from_stack(stack: OrderflowNautilusConfig) -> OrderflowStrategyConfig:
    """Convert OrderflowNautilusConfig to OrderflowStrategyConfig."""
    return OrderflowStrategyConfig(
        instrument_id=stack.instrument_id,
        order_id_tag=stack.order_id_tag,
        client_id=stack.client_id,
        timeframe=stack.timeframe,
        lookback_candles=stack.lookback_candles,
        book_depth=stack.book_depth,
        book_type=stack.book_type,
        eval_throttle_ms=stack.eval_throttle_ms,
        require_orderbook=stack.require_orderbook,
        # Map signals config
        signals_config=stack.signals_config if stack.signals_config else SignalsConfig(),
        # Map signal params
        imbalance_threshold=stack.signal.imbalance_threshold if stack.signal else 0.25,
        cvd_smoothing=stack.signal.cvd_smoothing if stack.signal else 5,
        absorption_min=stack.signal.absorption_min if stack.signal else 0.15,
        stack_min_rows=stack.signal.stack_min_rows if stack.signal else 3,
        # Map risk params
        max_position_fraction=stack.risk.max_position_fraction if stack.risk else 0.10,
        max_notional_usdt=stack.risk.max_notional_usdt if stack.risk else None,
        max_leverage=stack.risk.max_leverage if stack.risk else None,
        max_daily_loss_pct=stack.risk.max_daily_loss_pct if stack.risk else 5.0,
        max_consecutive_losses=stack.risk.max_consecutive_losses if stack.risk else 5,
        max_spread_bps=stack.risk.max_spread_bps if stack.risk else 25.0,
        stale_tick_ms=stack.risk.stale_tick_ms if stack.risk else 5000.0,
        min_top_of_book_qty=stack.risk.min_top_of_book_qty if stack.risk else 0.0,
        loss_cooldown_secs=stack.risk.loss_cooldown_secs if stack.risk else 0.0,
        # NOTE: min_hold_secs lives under execution in YAML profiles; fall back
        # to risk.min_hold_secs for backward compatibility.
        min_hold_secs=(
            stack.execution.min_hold_secs
            if (stack.execution and getattr(stack.execution, "min_hold_secs", None) is not None)
            else (stack.risk.min_hold_secs if stack.risk else 0.0)
        ),
        kill_switch_path=stack.risk.kill_switch_path if stack.risk else None,
        force_exit_path=stack.risk.force_exit_path if stack.risk else "orderflow/.force_exit",
        risk_per_trade_pct=stack.risk.risk_per_trade_pct if stack.risk else 0.0025,
        min_stop_bps=stack.risk.min_stop_bps if stack.risk else 12.0,
        max_stop_bps=stack.risk.max_stop_bps if stack.risk else 250.0,
        # Map execution params
        use_market_entries=stack.execution.use_market_entries if stack.execution else False,
        entry_post_only=stack.execution.entry_post_only if stack.execution else True,
        stoploss_pct=stack.execution.stoploss_pct if stack.execution else 0.02,
        trailing_trigger_pct=stack.execution.trailing_trigger_pct if stack.execution else 0.015,
        trailing_offset_pct=stack.execution.trailing_offset_pct if stack.execution else 0.01,
        max_time_in_trade_secs=stack.execution.max_time_in_trade_secs if stack.execution else None,
        max_entry_drift_bps=stack.execution.max_entry_drift_bps if stack.execution else 8.0,
        backtest_fee_rate=stack.execution.backtest_fee_rate if stack.execution else 0.0004,
        backtest_slippage_rate=stack.execution.backtest_slippage_rate if stack.execution else 0.0002,
        min_structural_rr=stack.execution.min_structural_rr if stack.execution else 1.2,
        max_structural_stop_bps=stack.execution.max_structural_stop_bps if stack.execution else 35.0,
        short_rejection_min_confidence=stack.execution.short_rejection_min_confidence if stack.execution else 1.05,
        short_rejection_size_mult=stack.execution.short_rejection_size_mult if stack.execution else 0.6,
        early_invalidation_secs=stack.execution.early_invalidation_secs if stack.execution else 120.0,
        early_invalidation_loss_pct=stack.execution.early_invalidation_loss_pct if stack.execution else 0.0018,
        acceptance_failure_secs=stack.execution.acceptance_failure_secs if stack.execution else 60.0,
        acceptance_failure_evals=stack.execution.acceptance_failure_evals if stack.execution else 2,
        # Map profile params
        vp_config=stack.vp_config if stack.vp_config else None,
        heatmap_config=stack.heatmap_config if stack.heatmap_config else None,
        divergence_window=stack.signal.divergence_window if stack.signal else 3,
        price_bucket_size=stack.signal.price_bucket_size if stack.signal else 1.0,
        large_trade_pct=stack.signal.large_trade_pct if stack.signal else 0.90,
    )
