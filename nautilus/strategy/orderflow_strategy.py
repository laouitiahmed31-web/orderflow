"""
strategy/orderflow_strategy.py — Heatmap-anchored bidirectional orderflow strategy.

Architecture changes vs previous version
-----------------------------------------
1. LiquidityHeatmap integrated: fed on every trade tick and OB snapshot.
   Signals now receive heatmap as 4th argument.

2. NoiseFilterStack: pre-signal gate that rejects low-quality market conditions.
   Prevents trading chop and mid-wave entries.

3. Wall-anchored exits: stop and target prices are set at entry time from the
   heatmap, not computed from arbitrary percentages.
   - long_stop  = just below nearest support wall
   - long_target = nearest resistance wall
   - short_stop = just above nearest resistance wall
   - short_target = nearest support wall
   These are STRUCTURAL levels, not %-based guesses.

4. Signal-reversal exits REMOVED. CVD/imbalance reversal checks are gone.
   They are lagging and were causing exits at the worst point of retracements.
   Exits are now: hard stop, wall break, target hit, trailing stop, time stop.

5. post-loss cooldown and min_hold_secs are ENABLED in config (not zeroed out).

6. _is_pending reset added to _exit_all (was missing — caused stuck state).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.core.data import Data
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas, TradeTick
from nautilus_trader.model.enums import OrderSide, PositionSide, book_type_from_str
from nautilus_trader.model.events import OrderFilled, PositionClosed
from nautilus_trader.model.identifiers import ClientId, InstrumentId
from nautilus_trader.trading.strategy import Strategy

from paper_trader import PaperTrader
from nautilus.config.schema import OrderflowStrategyConfig, SignalsConfig
from nautilus.data.ticks import trade_tick_to_side_dict
from nautilus.execution.policy import (
    BracketSpec,
    build_entry_order,
    build_exit_order,
    compute_bracket_prices,
    estimate_order_qty,
    estimate_order_qty_from_risk,
    should_cancel_stale_limit,
)
from nautilus.features.heatmap import LiquidityHeatmap, HeatmapSnapshot
from nautilus.features.multi_tf import MultiTFEngine, MultiTFSnapshot
from nautilus.features.ob import orderbook_to_imbalance
from nautilus.ml.dataset import DatasetBuffer, FeatureRow, Labeler, PassthroughHook
from nautilus.ops.metrics import MetricsLogger
from nautilus.risk.stack import PreTradeRiskStack
from nautilus.sessions.filter import SessionFilter
from nautilus.signals.base import EntrySignal
from nautilus.signals.filters import NoiseFilterStack
from nautilus.signals.registry import SignalRegistry
from nautilus.structure.market_structure import (
    NULL_STRUCTURE,
    MarketStructureEngine,
    MarketStructureSnapshot,
)


class OrderflowStrategy(Strategy):
    """
    Heatmap-anchored bidirectional orderflow strategy.

    Entry: only at heatmap walls, with absorption starting.
    Exit:  structural (wall-anchored), never lagging signal reversal.

    State machine:
        flat ──► LONG  (wall absorption / divergence / breakout)
        flat ──► SHORT (wall absorption / divergence / breakout)
        LONG  ──► flat (wall break stop / target hit / trailing / time)
        SHORT ──► flat (wall break stop / target hit / trailing / time)
    """

    def __init__(self, config: OrderflowStrategyConfig) -> None:
        super().__init__(config)
        self._instrument_id: InstrumentId = config.instrument_id
        self._client_id = ClientId(config.client_id)

        self._engine = MultiTFEngine(
            ltf=config.timeframe,
            htf=getattr(config, "htf_timeframe", "1h"),
            lookback_candles=config.lookback_candles,
            price_bucket_size=config.price_bucket_size,
            large_trade_pct=config.large_trade_pct,
            cvd_smoothing=config.cvd_smoothing,
            divergence_window=getattr(config, "divergence_window", 3),
        )

        # ── Heatmap ───────────────────────────────────────────────────────
        hm_cfg = getattr(config, "heatmap_config", None) or {}
        self._heatmap_engine = LiquidityHeatmap(
            bucket_size=hm_cfg.get("bucket_size", config.price_bucket_size * 5),
            window_trades=hm_cfg.get("window_trades", 8_000),
            wall_percentile=hm_cfg.get("wall_percentile", 0.80),
            proximity_bps=hm_cfg.get("proximity_bps", 15.0),
            min_walls=hm_cfg.get("min_walls", 2),
            ob_weight=hm_cfg.get("ob_weight", 1.5),
            stop_buffer_bps=hm_cfg.get("stop_buffer_bps", 5.0),
        )
        self._heatmap: HeatmapSnapshot | None = None

        # ── Volume Profile ────────────────────────────────────────────────
        from nautilus.features.volume_profile import VolumeProfile
        vp_cfg = getattr(config, "vp_config", None) or {}
        self._poc_band_bps = float(vp_cfg.get("poc_band_bps", 8.0))
        self._va_band_bps = float(vp_cfg.get("va_band_bps", 10.0))
        self._vp_engine = VolumeProfile(
            bucket_size=vp_cfg.get("bucket_size", 10.0),
            window_trades=vp_cfg.get("window_trades", 8_000),
            value_area_pct=vp_cfg.get("value_area_pct", 0.70),
            hvn_percentile=vp_cfg.get("hvn_percentile", 0.75),
            lvn_percentile=vp_cfg.get("lvn_percentile", 0.25),
            proximity_bps=vp_cfg.get("proximity_bps", 15.0),
            min_buckets=vp_cfg.get("min_buckets", 10),
            stop_buffer_bps=vp_cfg.get("stop_buffer_bps", 5.0),
            poc_band_bps=vp_cfg.get("poc_band_bps", 8.0),
            va_band_bps=vp_cfg.get("va_band_bps", 10.0),
            session_mode=vp_cfg.get("session_mode", False),
        )
        self._vp: VolumeProfileSnapshot | None = None

        # ── Noise filter stack ────────────────────────────────────────────
        self._noise_filter = NoiseFilterStack.default()

        # ── Signals ───────────────────────────────────────────────────────
        signals_cfg = getattr(config, "signals_config", None)
        if signals_cfg is None:
            signals_cfg = SignalsConfig(
                long=["hvn_absorption_long", "hvn_divergence_long", "poc_reclaim_long"],
                short=["hvn_absorption_short", "hvn_divergence_short", "poc_rejection_short"],
            )
        self._signals = SignalRegistry.from_config(signals_cfg)

        self._structure_engine = MarketStructureEngine(
            swing_window=getattr(config, "swing_window", 5)
        )
        self._structure: MarketStructureSnapshot = NULL_STRUCTURE

        sessions_cfg = getattr(config, "sessions_config", None)
        if sessions_cfg:
            self._session_filter = SessionFilter.from_config(sessions_cfg)
        else:
            self._session_filter = SessionFilter.always()

        self._risk = PreTradeRiskStack(
            max_daily_loss_pct=config.max_daily_loss_pct,
            max_consecutive_losses=config.max_consecutive_losses,
            max_spread_bps=config.max_spread_bps,
            stale_tick_ms=config.stale_tick_ms,
            min_top_of_book_qty=config.min_top_of_book_qty,
            kill_switch_path=config.kill_switch_path,
            max_leverage=config.max_leverage,
            equity_state_path=config.equity_state_path,
        )

        # ── Paper Trading (automatic interception) ─────────────────────────
        self._paper_trader = PaperTrader()
        self._paper_mode = True  # Always enabled for safety
        self._last_price: float = 0.0

        self._inference_hook = PassthroughHook()
        self._dataset = DatasetBuffer(labeler=Labeler()) if config.log_metrics else None
        self._metrics = MetricsLogger(config.metrics_dir) if config.log_metrics else None
        fe = getattr(config, "force_exit_path", None)
        self._force_exit_path: Path | None = Path(fe) if fe else None

        # ── Position state ────────────────────────────────────────────────
        self._last_tick_ns: int = 0
        self._last_eval_ns: int = 0
        self._entry_price: float | None = None
        self._entry_side: OrderSide | None = None
        self._trailing_active: bool = False
        self._trailing_peak: float = 0.0
        self._position_open_ts_ns: int | None = None
        self._last_signal: EntrySignal | None = None
        self._pending_limit_price: float | None = None
        self._is_pending: bool = False
        
        # Track VP warmup for logging
        self._vp_was_cold: bool = True
        self._tick_count: int = 0

        # Wall-anchored bracket prices (set at entry, drive exits)
        self._entry_stop_price: float | None = None    # structural stop
        self._entry_target_price: float | None = None  # structural target

        # Acceptance-failure exits (for breakout/acceptance entries)
        self._accept_level_price: float | None = None
        self._accept_band_bps: float | None = None
        self._accept_fail_count: int = 0
        self._accept_fail_needed: int = int(getattr(config, "acceptance_failure_evals", 2))
        self._accept_fail_ns: int = 0
        self._accept_fail_last_ns: int | None = None
        self._accept_fail_needed_secs: float = float(getattr(config, "acceptance_failure_secs", 60.0))

        # ── Cooldown state ─────────────────────────────────────────────────
        self._loss_cooldown_until_ns: int = 0

        self._bracket = BracketSpec(
            stoploss_pct=config.stoploss_pct,
            target_pct=getattr(config, "target_pct", config.stoploss_pct * 2),
            trailing_trigger_pct=config.trailing_trigger_pct,
            trailing_offset_pct=config.trailing_offset_pct,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_start(self) -> None:
        self.log.info(f"=== on_start() called for {self._instrument_id} ===")
        vp_cfg = getattr(self.config, "vp_config", None) or {}
        self.log.info(f"[VP] Config: window_trades={vp_cfg.get('window_trades', 8000)}, min_buckets={vp_cfg.get('min_buckets', 10)}, bucket_size={vp_cfg.get('bucket_size', 10.0)}")
        
        # Initialize last_tick_ns to now so stale_tick check doesn't fail immediately
        self._last_tick_ns = self.clock.timestamp_ns()
        self.log.debug(f"[INIT] _last_tick_ns initialized to {self._last_tick_ns}")
        
        inst = self.cache.instrument(self._instrument_id)
        if inst is None:
            self.log.error(f"Instrument not found: {self._instrument_id}")
            self.stop()
            return

        positions = self.cache.positions_open(
            instrument_id=self._instrument_id, strategy_id=self.id
        )
        if positions:
            pos = positions[0]
            self.log.info(f"Recovered open position: {pos.side} {pos.quantity}")
            self._entry_price = float(pos.avg_px_open)
            self._entry_side = OrderSide.BUY if pos.is_long else OrderSide.SELL
            self._position_open_ts_ns = pos.ts_opened
            self._trailing_peak = self._entry_price

        self.subscribe_trade_ticks(self._instrument_id)
        bt = (
            book_type_from_str(self.config.book_type)
            if isinstance(self.config.book_type, str)
            else self.config.book_type
        )
        self.subscribe_order_book_deltas(self._instrument_id, bt)
        self.log.info(f"=== Subscribed to trade ticks and order book deltas ===")
        self.log.info(f"[READY] Strategy initialized and ready for signals")

    # ── Data handlers ──────────────────────────────────────────────────────────

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._tick_count += 1
        self._last_tick_ns = tick.ts_event
        self._last_price = float(tick.price)  # Track for paper trading
        
        raw = trade_tick_to_side_dict(tick)
        if raw is None:
            return

        # Keep tick logging sparse for fast backtests.
        if self._tick_count % 500_000 == 0:
            self.log.info(
                f"[TICK] processed={self._tick_count:,} "
                f"last={tick.price}@{tick.size} vp_buckets={self._vp_engine.bucket_count}"
            )
        self._engine.add_tick(raw["ts"], raw["price"], raw["qty"], raw["side"])

        # Feed heatmap on every trade
        self._heatmap_engine.add_trade(raw["price"], raw["qty"])
        
        # Feed volume profile on every trade
        if self._vp_engine:
            self._vp_engine.add_trade(raw["price"], raw["qty"])

        self._maybe_evaluate()

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        # Update last activity timestamp for stale tick check
        self._last_tick_ns = self.clock.timestamp_ns()
        try:
            # Update heatmap resting volume on OB deltas
            book = self.cache.order_book(self._instrument_id)
            if book:
                depth = getattr(self.config, "book_depth", 20)
                bids = [(float(level.price), float(level.size())) for level in list(book.bids())[:depth]]
                asks = [(float(level.price), float(level.size())) for level in list(book.asks())[:depth]]
                self._heatmap_engine.add_ob_snapshot(bids, asks)

            self._maybe_evaluate()
        except Exception as e:
            self.log.exception(f"Error in on_order_book_deltas: {e}", e)

    # ── Evaluation throttle ────────────────────────────────────────────────────

    def _maybe_evaluate(self) -> None:
        is_long  = self.portfolio.is_net_long(self._instrument_id)
        is_short = self.portfolio.is_net_short(self._instrument_id)

        throttle_ms = 50 if (is_long or is_short) else self.config.eval_throttle_ms
        throttle_ns = int(throttle_ms * 1_000_000)
        now_ns = self.clock.timestamp_ns()
        if throttle_ns > 0 and (now_ns - self._last_eval_ns) < throttle_ns:
            return
        self._last_eval_ns = now_ns
        self.log.debug(f"[EVAL] Evaluating market | is_long={is_long} is_short={is_short}")

        book = self.cache.order_book(self._instrument_id)
        ob_imb = orderbook_to_imbalance(book, self.config.book_depth)
        self._engine.set_orderbook_imbalance_value(ob_imb)

        now_ms = int(self._last_tick_ns / 1_000_000 if self._last_tick_ns else now_ns / 1_000_000)
        snap = self._engine.compute_snapshot(now_ms=now_ms)
        if snap is None:
            return

        # Refresh heatmap snapshot
        mid = book.midpoint() if book and book.midpoint() is not None else None
        px  = float(mid) if mid is not None else snap.ltf.close_price
        if px and self._heatmap_engine.is_warm:
            self._heatmap = self._heatmap_engine.compute_snapshot(px)
        else:
            self._heatmap = None

        # Refresh volume profile snapshot
        if px and self._vp_engine.is_warm:
            self._vp = self._vp_engine.compute_snapshot(px)
            if self._vp_was_cold:
                self.log.info(f"[VP] WARMED UP! {self._vp_engine.bucket_count} buckets, total_volume={self._vp_engine.total_volume:.0f}")
                self._vp_was_cold = False
        else:
            self._vp = None
            vp_buckets = self._vp_engine.bucket_count
            vp_min = getattr(self.config, "vp_config", {}).get("min_buckets", 10)
            if vp_buckets > 0 and vp_buckets % 100 == 0:  # Log every 100 buckets
                self.log.info(f"[VP] Warming up... {vp_buckets}/{vp_min} buckets")

        htf_candles = self._engine.completed_candles(
            getattr(self.config, "htf_timeframe", "1h")
        )
        if htf_candles and snap.htf:
            self._structure = self._structure_engine.update(htf_candles, snap.htf.close_price)

        session = self._session_filter.current_session(datetime.now(UTC))
        inst    = self.cache.instrument(self._instrument_id)
        if inst is None:
            return

        if self._metrics:
            self._metrics.log_event("market_update", {
                "price": px,
                "cvd_trend": "rising" if snap.ltf.cvd_rising else "falling",
                "ob_imbalance": float(ob_imb),
                "at_support": self._heatmap.at_support if self._heatmap else None,
                "at_resistance": self._heatmap.at_resistance if self._heatmap else None,
                "wall_strength": self._heatmap.wall_strength if self._heatmap else None,
                "heatmap_warm": self._heatmap_engine.is_warm,
            })

        if self._force_exit_path is not None and self._force_exit_path.exists():
            if is_long or is_short:
                pos_side = OrderSide.BUY if is_long else OrderSide.SELL
                self.log.warning("force_exit file — closing position")
                self._exit_all("operator_force_exit", pos_side)
                try:
                    self._force_exit_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            try:
                self._force_exit_path.unlink(missing_ok=True)
            except OSError:
                pass

        if is_long:
            self._check_exit(snap, px, OrderSide.BUY)
        elif is_short:
            self._check_exit(snap, px, OrderSide.SELL)
        else:
            self._maybe_cancel_replace_limit(px)
            self._check_entry(snap, book, px, session)

    # ── Account helpers ────────────────────────────────────────────────────────

    def _quote_balance(self) -> float | None:
        try:
            inst = self.cache.instrument(self._instrument_id)
            if inst is None:
                return None
            account = self.portfolio.account(self._instrument_id.venue)
            bal = account.balance(inst.quote_currency)
            return float(bal.total.as_double()) if bal else None
        except Exception:
            return None

    # ── Entry ──────────────────────────────────────────────────────────────────

    def _check_entry(self, snap: MultiTFSnapshot, book, px: float, session) -> None:
        cfg = self.config

        if self._is_pending:
            return

        now_ns = self.clock.timestamp_ns()
        if now_ns < self._loss_cooldown_until_ns:
            remaining_s = (self._loss_cooldown_until_ns - now_ns) / 1e9
            if self._metrics:
                self._metrics.log_event("entry_rejected", {
                    "failed": ["loss_cooldown"],
                    "remaining_s": remaining_s,
                })
            return

        if self.cache.orders_open_count(instrument_id=self._instrument_id, strategy_id=self.id) > 0:
            return

        inst = self.cache.instrument(self._instrument_id)
        if inst is None:
            return

        # ── Pre-trade risk gates ───────────────────────────────────────────
        for check_name, result in (
            ("kill_switch",  self._risk.check_kill_switch()),
            ("stale_tick",   self._risk.check_stale_tick(
                self._last_tick_ns / 1_000_000,
                now_ns / 1_000_000,
            )),
        ):
            if not result.ok:
                if self._metrics:
                    self._metrics.log_event("entry_rejected", {"failed": [check_name]})
                return

        if cfg.require_orderbook:
            if not book:
                return
            sd = self._risk.check_spread_and_depth(book)
            if not sd.ok:
                if self._metrics:
                    self._metrics.log_event("entry_rejected", {"failed": ["spread_depth"]})
                return

        if not self.portfolio.is_flat(self._instrument_id):
            return

        eq = self._quote_balance()
        if not self._risk.check_daily_loss(eq).ok:
            return

        # ── Evaluate signals (long and short in parallel) ─────────────────
        # Run noise filter for each direction, then evaluate signals.
        
        # Status check: require heatmap to be warm (structural stops must be reliable)
        if not self._heatmap_engine.is_warm:
            hm_tape = self._heatmap_engine.tape_length
            hm_min = int(self._heatmap_engine._tape.maxlen * 0.20) if hasattr(self._heatmap_engine, '_tape') else 1000
            if hm_tape % 1000 == 0 and hm_tape > 0:  # Log every 1000 tapes
                self.log.debug(f"[STATUS] Heatmap warming: {hm_tape}/{hm_min} tapes (20% threshold)")
            return  # Can't process signals without warm heatmap for structural stops
        
        long_signal  = self._evaluate_direction(snap, session, is_long=True)
        short_signal = self._evaluate_direction(snap, session, is_long=False)

        # FIX 11: if both directions fire, pick the higher-confidence one.
        # If tied, skip entirely — ambiguous market means no trade.
        if long_signal and short_signal:
            if long_signal.confidence > short_signal.confidence:
                signal = long_signal
            elif short_signal.confidence > long_signal.confidence:
                signal = short_signal
            else:
                if self._metrics:
                    self._metrics.log_event("entry_rejected", {"failed": ["ambiguous_direction"]})
                return
        else:
            signal = long_signal or short_signal
        if signal is None:
            return

        if eq is None or eq <= 0:
            return

        ml_confidence = float(
            self._inference_hook.predict(self._build_feature_row(snap, session, signal))
        )
        if ml_confidence <= 0:
            return

        # Combine model confidence with signal module confidence for sizing.
        sig_conf = float(getattr(signal, "confidence", 1.0))
        combined_conf = max(0.0, min(1.0, ml_confidence)) * max(0.0, min(1.0, sig_conf))

        # ── Store structural bracket prices (HEATMAP stops + VP targets) ──
        # FIX: Use HEATMAP for stops (real structural support/resistance walls)
        # Use VP only for acceptance entry targets (POC/VAH/VAL objectives)
        # This prevents being stopped in the value area noise.
        if self._heatmap is not None:
            if signal.side == OrderSide.BUY:
                # Structural stop from heatmap wall
                self._entry_stop_price = self._heatmap.long_stop_price
                # Target: use VP for acceptance trades, heatmap otherwise
                if "acceptance" in signal.label and self._vp is not None and self._vp.is_valid:
                    candidates = [
                        self._vp.long_target_price,
                        self._vp.long_travel_target_price,
                        self._vp.vah_price,
                    ]
                    candidates = [c for c in candidates if c is not None and c > px]
                    self._entry_target_price = max(candidates) if candidates else self._heatmap.long_target_price
                else:
                    self._entry_target_price = self._heatmap.long_target_price
            else:  # SHORT
                # Structural stop from heatmap wall
                self._entry_stop_price = self._heatmap.short_stop_price
                # Target: use VP for acceptance trades, heatmap otherwise
                if "acceptance" in signal.label and self._vp is not None and self._vp.is_valid:
                    candidates = [
                        self._vp.short_target_price,
                        self._vp.short_travel_target_price,
                        self._vp.val_price,
                    ]
                    candidates = [c for c in candidates if c is not None and c < px]
                    self._entry_target_price = min(candidates) if candidates else self._heatmap.short_target_price
                else:
                    self._entry_target_price = self._heatmap.short_target_price
        else:
            self._entry_stop_price   = None
            self._entry_target_price = None

        # Reject entries without structural stop (risk sizing requires it)
        if self._entry_stop_price is None:
            if self._metrics:
                self._metrics.log_event("entry_rejected", {"failed": ["no_structural_stop"]})
            return

        # Stop distance gates (institutional noise/tail control)
        stop_bps = abs(px - self._entry_stop_price) / px * 10_000.0
        min_stop_bps = float(getattr(cfg, "min_stop_bps", 0.0))
        max_stop_bps = float(getattr(cfg, "max_stop_bps", 10_000.0))
        if stop_bps < min_stop_bps or stop_bps > max_stop_bps:
            if self._metrics:
                self._metrics.log_event("entry_rejected", {"failed": ["stop_bps_out_of_bounds"]})
            return

        # Risk-based sizing with hard clamps
        # Use fake balance for paper trading so quantity validation doesn't fail
        qty_calc_balance = 10000.0 if self._paper_mode else eq
        qty = estimate_order_qty_from_risk(
            inst,
            equity=qty_calc_balance,
            entry_price=px,
            stop_price=float(self._entry_stop_price),
            risk_per_trade_pct=float(getattr(cfg, "risk_per_trade_pct", 0.0)) * combined_conf,
            max_fraction=float(getattr(cfg, "max_position_fraction", 0.0)),
            max_notional_usdt=getattr(cfg, "max_notional_usdt", None),
        )

        if qty <= 0:
            if self._metrics:
                self._metrics.log_event("entry_rejected", {"failed": ["zero_qty_risk_sizer"]})
            return

        notional = float(qty) * px
        if not self._risk.check_leverage(notional, eq).ok:
            return

        try:
            order = build_entry_order(
                self.order_factory,
                inst,
                side=signal.side,
                price=px,
                qty=qty,
                use_market=cfg.use_market_entries,
                post_only=cfg.entry_post_only,
            )
        except Exception as exc:
            self.log.warning(f"Entry order build failed: {exc}")
            return

        # Geometry validation - reject entries with excessive stop distances
        if self._entry_stop_price is not None and self._entry_target_price is not None:
            if signal.side == OrderSide.BUY:
                risk = (px - self._entry_stop_price) / px
                reward = (self._entry_target_price - px) / px
            else:  # SELL
                risk = (self._entry_stop_price - px) / px
                reward = (px - self._entry_target_price) / px
            
            if risk <= 0.0 or reward <= 0.0:
                self.log.warning(f"[ENTRY] Invalid geometry: risk={risk:.4f}, reward={reward:.4f}")
                return
            
            rr = reward / risk
            max_stop_bps = getattr(cfg, "max_structural_stop_bps", 35.0)
            min_rr = getattr(cfg, "min_structural_rr", 1.2)
            risk_bps = risk * 10_000.0
            
            self.log.info(f"[ENTRY] Geometry check: risk={risk_bps:.1f}bps, max={max_stop_bps:.1f}bps, rr={rr:.2f}, min={min_rr:.2f}")
            if risk_bps > max_stop_bps or rr < min_rr:
                self.log.warning(f"[ENTRY] Geometry rejected: risk={risk_bps:.1f}bps > {max_stop_bps:.1f}bps, rr={rr:.2f} < {min_rr:.2f}")
                return

        self._entry_price         = px
        self._entry_side          = signal.side
        self._trailing_active     = False
        self._trailing_peak       = px
        self._position_open_ts_ns = self.clock.timestamp_ns()
        self._last_signal         = signal
        self._pending_limit_price = px if not cfg.use_market_entries else None
        # Configure acceptance failure level based on entry type.
        self._accept_level_price = None
        self._accept_band_bps = None
        self._accept_fail_count = 0
        if self._vp is not None and self._vp.is_valid:
            if signal.label in ("poc_acceptance_retest_long", "poc_acceptance_retest_short"):
                self._accept_level_price = self._vp.poc_price
                self._accept_band_bps = self._poc_band_bps
            elif signal.label == "vah_acceptance_long":
                self._accept_level_price = self._vp.vah_price
                self._accept_band_bps = self._va_band_bps
            elif signal.label == "val_acceptance_short":
                self._accept_level_price = self._vp.val_price
                self._accept_band_bps = self._va_band_bps

        self.log.info(
            f"[SIGNAL] {signal.side.name} @ {px:.2f} | {signal.label} "
            f"(conf={combined_conf:.3f}) qty={qty:.4f} stop_bps={stop_bps:.1f}"
        )
        self.submit_order(order, client_id=self._client_id)
        self._is_pending = True

        if self._metrics:
            self._metrics.log_event("entry_signal", {
                "side":          signal.side.name,
                "label":         signal.label,
                "price":         px,
                "qty":           str(qty),
                "notional_usdt": float(qty) * px,
                "confidence":    ml_confidence,
                "conditions":    signal.conditions,
                "stop_price":    self._entry_stop_price,
                "target_price":  self._entry_target_price,
                "wall_strength": self._heatmap.wall_strength if self._heatmap else None,
            })

    def _evaluate_direction(
        self,
        snap: MultiTFSnapshot,
        session,
        is_long: bool,
    ) -> EntrySignal | None:
        """Run noise filter then signal evaluation for one direction."""
        direction = "LONG" if is_long else "SHORT"
        noise = self._noise_filter.check(snap, self._vp, session, is_long=is_long)
        if not noise.passed:
            self.log.debug(f"[{direction}] Noise filter blocked: {noise.failed_filters}")
            if self._metrics:
                self._metrics.log_event("noise_filter_block", {
                    "direction": "long" if is_long else "short",
                    "failed": noise.failed_filters,
                })
            return None

        # FIX 1: pass vp= so every signal module receives the VP snapshot
        signals = (
            self._signals.evaluate_long(snap, self._structure, session, vp=self._vp)
            if is_long
            else self._signals.evaluate_short(snap, self._structure, session, vp=self._vp)
        )
        signal = None
        if signals:
            # Prefer higher-confidence modules; tie-break by richer condition set.
            signal = max(signals, key=lambda s: (float(s.confidence), len(s.conditions)))

        # ── Regime bias gate (institutional: trade with HTF structure) ──────────
        # This prevents counter-trend shorts in bull runs (and vice versa) which
        # are typically just noise stops in trend.
        if signal is not None:
            trend = getattr(self._structure, "trend", None)
            trend_val = getattr(trend, "value", None)
            if signal.side == OrderSide.SELL:
                # Only short in bearish regime, or on active breakdown BOS.
                if trend_val != "bearish":
                    if not (self._structure.structure_break and self._structure.break_type == "low"):
                        return None
            if signal.side == OrderSide.BUY:
                # Only long in bullish regime, or on active upside BOS.
                if trend_val != "bullish":
                    if not (self._structure.structure_break and self._structure.break_type == "high"):
                        return None
        if signal:
            self.log.debug(
                f"[{direction}] Signal fired: {signal.label} "
                f"(conf={signal.confidence:.3f}, candidates={len(signals)})"
            )
        return signal

    # ── Exit ───────────────────────────────────────────────────────────────────

    def _check_exit(self, snap: MultiTFSnapshot, px: float, position_side: OrderSide) -> None:
        """
        Wall-anchored exit logic.

        Priority order:
          1. Hard stop (percentage-based emergency backstop)
          2. Structural stop (wall break — thesis invalidated)
          3. Minimum hold time (suppress exits for min_hold_secs after entry)
          4. Profit target (structural — at next wall)
          5. Trailing stop
          6. Time stop
          
        Signal-reversal exits (CVD/imbalance) have been REMOVED.
        They were lagging, causing exits at the bottom of retracements.
        Wall-anchored exits provide structural levels that don't lag.
        """
        cfg = self.config
        if self._entry_price is None:
            return

        entry     = self._entry_price
        direction = 1.0 if position_side == OrderSide.BUY else -1.0
        pnl_pct   = direction * (px - entry) / entry

        # ── 1. Hard stop (emergency backstop — fires regardless of hold time) ─
        stop_dist = self._bracket.stoploss_pct
        if pnl_pct <= -stop_dist:
            self._exit_all("hard_stop", position_side)
            return

        # ── 2. Acceptance-failure exit (breakout entries) ──────────────────
        # For breakout/acceptance trades, the thesis fails when price persists
        # back through the acceptance boundary (POC/VAH/VAL band), not on a 1-tick
        # probe into the structural stop zone.
        if self._accept_level_price is not None and self._accept_band_bps is not None:
            band = self._accept_band_bps / 10_000.0
            if position_side == OrderSide.BUY:
                failed = px < (self._accept_level_price * (1.0 - band))
            else:
                failed = px > (self._accept_level_price * (1.0 + band))
            if failed:
                now_ns = self.clock.timestamp_ns()
                if self._accept_fail_last_ns is None:
                    self._accept_fail_last_ns = now_ns
                dt = max(0, now_ns - self._accept_fail_last_ns)
                self._accept_fail_last_ns = now_ns
                self._accept_fail_ns += dt
                self._accept_fail_count += 1
                if (
                    self._accept_fail_ns >= int(max(0.0, self._accept_fail_needed_secs) * 1e9)
                    or self._accept_fail_count >= max(1, self._accept_fail_needed)
                ):
                    self._exit_all("acceptance_fail", position_side)
                    return
            else:
                self._accept_fail_count = 0
                self._accept_fail_ns = 0
                self._accept_fail_last_ns = None

        # ── 3. Structural stop (disaster line) ─────────────────────────────
        # If price hard breaks through the structural stop, exit immediately.
        if self._entry_stop_price is not None:
            if position_side == OrderSide.BUY and px <= self._entry_stop_price:
                self._exit_all("wall_break_stop", position_side)
                return
            if position_side == OrderSide.SELL and px >= self._entry_stop_price:
                self._exit_all("wall_break_stop", position_side)
                return

        # ── 4. Minimum hold time (suppress further exits until elapsed) ────
        min_hold_secs = getattr(cfg, "min_hold_secs", 10.0)
        if self._position_open_ts_ns is not None and min_hold_secs > 0:
            held_secs = (self.clock.timestamp_ns() - self._position_open_ts_ns) / 1e9
            if held_secs < min_hold_secs:
                return

        # ── 5. Structural profit target (next wall) ─────────────────────────
        if self._entry_target_price is not None:
            if position_side == OrderSide.BUY and px >= self._entry_target_price:
                # Acceptance trades should not fully exit at the first objective.
                # Convert objective hit into "runner mode": activate trailing and
                # remove the target so we don't scalp tiny wins.
                if self._last_signal and "acceptance" in self._last_signal.label:
                    self._trailing_active = True
                    self._trailing_peak = max(self._trailing_peak, px)
                    self._entry_target_price = None
                    if self._metrics:
                        self._metrics.log_event("target_to_trailing", {"side": "BUY", "px": px})
                    return
                self._exit_all("wall_target_hit", position_side)
                return
            if position_side == OrderSide.SELL and px <= self._entry_target_price:
                if self._last_signal and "acceptance" in self._last_signal.label:
                    self._trailing_active = True
                    self._trailing_peak = min(self._trailing_peak, px)
                    self._entry_target_price = None
                    if self._metrics:
                        self._metrics.log_event("target_to_trailing", {"side": "SELL", "px": px})
                    return
                self._exit_all("wall_target_hit", position_side)
                return

        # ── 6. Trailing stop ─────────────────────────────────────────────────
        if pnl_pct >= self._bracket.trailing_trigger_pct:
            self._trailing_active = True

        if self._trailing_active:
            if position_side == OrderSide.BUY:
                self._trailing_peak = max(self._trailing_peak, px)
                trail_dd = (px - self._trailing_peak) / self._trailing_peak
            else:
                self._trailing_peak = min(self._trailing_peak, px)
                trail_dd = (self._trailing_peak - px) / self._trailing_peak

            if trail_dd <= -self._bracket.trailing_offset_pct:
                self._exit_all("trailing_stop", position_side)
                return

        # ── 7. Time stop ─────────────────────────────────────────────────────
        if cfg.max_time_in_trade_secs is not None and self._position_open_ts_ns is not None:
            open_secs = (self.clock.timestamp_ns() - self._position_open_ts_ns) / 1e9
            if open_secs >= cfg.max_time_in_trade_secs:
                self._exit_all("time_stop", position_side)
                return

    # ── Exit execution ─────────────────────────────────────────────────────────

    def _exit_all(self, reason: str, side: OrderSide) -> None:
        self.cancel_all_orders(self._instrument_id, client_id=self._client_id)
        positions = self.cache.positions_open(
            instrument_id=self._instrument_id,
            strategy_id=self.id,
        )
        for pos in positions:
            self.close_position(pos, client_id=self._client_id, tags=[f"exit:{reason}"])

        if self._metrics:
            self._metrics.log_event("exit", {
                "reason":           reason,
                "side":             side.name,
                "entry_price":      self._entry_price,
                "stop_price":       self._entry_stop_price,
                "target_price":     self._entry_target_price,
                "trailing_active":  self._trailing_active,
            })

        self._entry_price         = None
        self._entry_side          = None
        self._trailing_active     = False
        self._position_open_ts_ns = None
        self._pending_limit_price = None
        self._entry_stop_price    = None
        self._entry_target_price  = None
        self._accept_level_price  = None
        self._accept_band_bps     = None
        self._accept_fail_count   = 0
        self._accept_fail_ns      = 0
        self._accept_fail_last_ns = None
        self._is_pending          = False   # FIX: was missing, caused stuck state

    # ── Cancel-replace stale limit ─────────────────────────────────────────────

    def _maybe_cancel_replace_limit(self, current_px: float) -> None:
        if self._pending_limit_price is None:
            return

        open_count = self.cache.orders_open_count(
            instrument_id=self._instrument_id,
            strategy_id=self.id,
        )
        if open_count == 0:
            self._pending_limit_price = None
            return

        if self._entry_side is None:
            self._pending_limit_price = None
            return

        if should_cancel_stale_limit(
            self._pending_limit_price,
            current_px,
            side=self._entry_side,
            max_drift_bps=getattr(self.config, "max_entry_drift_bps", 8.0),
        ):
            stale_px = self._pending_limit_price
            self.cancel_all_orders(self._instrument_id, client_id=self._client_id)
            self._pending_limit_price = None
            self._entry_price = None
            if self._metrics:
                self._metrics.log_event("entry_cancelled_stale_limit", {
                    "limit_px":   stale_px,
                    "current_px": current_px,
                })

    # ── Order feedback ─────────────────────────────────────────────────────────

    def on_order_submitted(self, event):
        pass

    def on_order_filled(self, event: OrderFilled) -> None:
        if event.instrument_id != self._instrument_id or not self._metrics:
            return
        px   = event.last_px
        qty  = event.last_qty
        comm = event.commission
        liq  = event.liquidity_side
        self._metrics.log_event("fill", {
            "side":          event.order_side.name,
            "price":         float(px.as_double()) if px is not None else None,
            "qty":           str(qty) if qty is not None else None,
            "trade_id":      str(event.trade_id),
            "liquidity_side": liq.name if hasattr(liq, "name") else str(liq),
            "fee":           float(comm.as_double()) if comm is not None else None,
        })

    def on_order_accepted(self, event):
        self._is_pending = False

    def on_position_opened(self, event):
        self._entry_price         = float(event.avg_px_open)
        self._entry_side          = OrderSide.BUY if event.side == PositionSide.LONG else OrderSide.SELL
        self._position_open_ts_ns = event.ts_event
        self._trailing_peak       = self._entry_price
        self._trailing_active     = False

    def on_order_rejected(self, event):
        self._is_pending   = False
        self._entry_price  = None
        self._entry_side   = None

    def on_order_cancelled(self, event):
        self._is_pending   = False
        self._entry_price  = None
        self._entry_side   = None

    # ── Position closed ────────────────────────────────────────────────────────

    def on_position_closed(self, event: PositionClosed) -> None:
        if event.instrument_id != self._instrument_id:
            return

        pnl = float(event.realized_pnl.as_double())
        self._risk.on_position_closed_pnl(pnl)

        if pnl < 0:
            cooldown_secs = getattr(self.config, "loss_cooldown_secs", 60.0)
            if cooldown_secs > 0:
                self._loss_cooldown_until_ns = (
                    self.clock.timestamp_ns() + int(cooldown_secs * 1e9)
                )
                self.log.info(f"Loss cooldown active for {cooldown_secs:.0f}s (pnl={pnl:.4f})")

        self._entry_price         = None
        self._entry_side          = None
        self._trailing_active     = False
        self._position_open_ts_ns = None
        self._entry_stop_price    = None
        self._entry_target_price  = None
        self._accept_level_price  = None
        self._accept_band_bps     = None
        self._accept_fail_count   = 0
        self._accept_fail_ns      = 0
        self._accept_fail_last_ns = None

        if self._metrics:
            eq = self._quote_balance()
            self._metrics.log_event("position_closed", {
                "realized_pnl":        pnl,
                "consecutive_losses":  self._risk.consecutive_losses,
                "daily_pnl_pct":       self._risk.daily_pnl_pct(eq),
            })

    # ── ML feature row ─────────────────────────────────────────────────────────

    def _build_feature_row(self, snap: MultiTFSnapshot, session, signal: EntrySignal) -> FeatureRow:
        ltf = snap.ltf.flow
        # FIX 13: snap.htf can be None during HTF warmup — guard all htf accesses
        htf_snap = snap.htf
        htf      = htf_snap.flow if htf_snap is not None else None
        st  = self._structure
        hm  = self._heatmap
        large_sum = ltf.large_buy_vol + ltf.large_sell_vol
        large_dom = (
            (ltf.large_buy_vol - ltf.large_sell_vol) / large_sum
            if large_sum > 1e-9 else 0.0
        )
        return FeatureRow(
            ts_ms=snap.ltf.ts_ms,
            cvd=ltf.cvd, cvd_ema=snap.ltf.cvd_ema, cvd_rising=int(snap.ltf.cvd_rising),
            imbalance=ltf.imbalance, absorption=ltf.absorption,
            stacked_imb=ltf.stacked_imb, ob_imbalance=ltf.ob_imbalance,
            delta_div=ltf.delta_div, large_dom=large_dom,
            buy_vol=ltf.buy_vol, sell_vol=ltf.sell_vol, total_vol=ltf.total_vol,
            # FIX 13: use safe defaults when HTF snapshot not yet available
            htf_cvd=htf.cvd if htf else 0.0,
            htf_cvd_rising=int(htf_snap.cvd_rising) if htf_snap else 0,
            htf_imbalance=htf.imbalance if htf else 0.0,
            htf_absorption=htf.absorption if htf else 0.0,
            trend=1 if st.trend.value == "bullish" else (-1 if st.trend.value == "bearish" else 0),
            hh=1 if st.last_swing_high and st.structure_break and st.break_type == "high" else 0,
            hl=1 if st.last_swing_low and st.structure_break else 0,
            lh=0, ll=0,
            bos_bullish=1 if st.structure_break and st.break_type == "high" else 0,
            bos_bearish=1 if st.structure_break and st.break_type == "low" else 0,
            last_high_price=st.last_swing_high.price if st.last_swing_high else 0.0,
            last_low_price=st.last_swing_low.price if st.last_swing_low else 0.0,
            session_name=session.session_name or "",
            session_active=int(session.active),
            session_minutes_elapsed=session.minutes_elapsed or -1,
            session_minutes_to_close=session.minutes_to_close or -1,
            signal_label=signal.label,
            signal_side=signal.side.name,
            signal_confidence=signal.confidence,
            # FIX 6: heatmap fields stored in the extended features dict,
            # not as top-level FeatureRow kwargs (those fields don't exist on the dataclass)
            features={
                "at_support":             int(hm.at_support) if hm else 0,
                "at_resistance":          int(hm.at_resistance) if hm else 0,
                "wall_strength":          hm.wall_strength if hm else 0.0,
                "nearest_support_dist":   hm.nearest_support.distance_bps if (hm and hm.nearest_support) else -1.0,
                "nearest_resistance_dist":hm.nearest_resistance.distance_bps if (hm and hm.nearest_resistance) else -1.0,
            },
        )

    def submit_order(self, order, *args, **kwargs) -> None:
        """Submit order - in paper mode, only simulate. In live mode, submit to exchange."""
        if self._paper_mode:
            # Paper mode: ONLY simulate, do NOT submit to real exchange
            side = "BUY" if order.side == OrderSide.BUY else "SELL"
            qty = float(order.quantity)
            usdt_amount = qty * self._last_price if side == "BUY" else 0
            
            result = self._paper_trader.place_order(
                symbol=self._instrument_id.symbol.value,
                side=side,
                usdt_amount=usdt_amount,
                current_price=self._last_price
            )
            
            if result.get("error"):
                self.log.error(f"[PAPER] Trade failed: {result['error']}")
            else:
                self.log.info(f"[PAPER] {side} {qty:.6f} @ ${self._last_price:.2f} | {result.get('status', 'OK')}")
            # Return WITHOUT submitting to real exchange
            return
        
        # Live mode: submit real order
        price = getattr(order, "price", None)
        if price is None:
            self.log.info(f"[LIVE] Submitting real order: {order.side} {order.quantity} @ MARKET")
        else:
            self.log.info(f"[LIVE] Submitting real order: {order.side} {order.quantity} @ {price}")
        super().submit_order(order, *args, **kwargs)

    def on_data(self, data: Data) -> None:
        pass