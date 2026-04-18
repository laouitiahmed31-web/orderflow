"""
signals/long.py — Volume-profile-anchored long entry signals.

Swap from heatmap: at_support/wall_strength → at_hvn_below/hvn.volume_pct
POC context added: above_poc = bullish bias, below_poc = counter-trend (tighter gates).
LVN awareness: divergence signal explicitly checks we're not at an LVN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from nautilus_trader.model.enums import OrderSide

from nautilus.signals.base import EntrySignal, SignalModule

if TYPE_CHECKING:
    from nautilus.features.volume_profile import VolumeProfileSnapshot
    from nautilus.features.multi_tf import MultiTFSnapshot
    from nautilus.sessions.filter import SessionState
    from nautilus.structure.market_structure import MarketStructureSnapshot


# ════════════════════════════════════════════════════════════════════════════════
#  1. HVN ABSORPTION LONG  ← PRIMARY SIGNAL
#
#  Enter at a High Volume Node below price (support) with absorption starting.
#  HVN = price level where significant volume has traded = structural defense.
#
#  If price is above POC: with-trend, normal thresholds.
#  If price is below POC: counter-trend, tighter gates.
# ════════════════════════════════════════════════════════════════════════════════

class HVNAbsorptionLong(SignalModule):
    label = "hvn_absorption_long"
    side  = OrderSide.BUY

    def __init__(
        self,
        absorption_min: float = 0.08,
        ob_imb_min: float = 0.05,
        min_hvn_volume_pct: float = 0.03,      # HVN must be meaningful
        max_bearish_stack: float = -3.0,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min    = absorption_min
        self._ob_min     = ob_imb_min
        self._min_hvn    = min_hvn_volume_pct
        self._max_bstack = max_bearish_stack
        self._htf_align  = require_htf_align

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid:
            return None

        of  = snap.ltf.flow
        hvn = vp.nearest_hvn_below

        # Tighter absorption threshold when trading against POC
        abs_min = self._abs_min if vp.above_poc else self._abs_min * 1.5

        conditions = {
            "at_hvn_support":     vp.at_hvn_below,
            "hvn_meaningful":     hvn is not None and hvn.volume_pct >= self._min_hvn,
            "not_at_lvn":         not vp.at_lvn,
            "hvn_target_exists":  vp.nearest_hvn_above is not None,
            "absorption_start":   of.absorption >= abs_min,
            "directional_signal": snap.ltf.cvd_rising or of.delta_div == -1.0,
            "ob_bid_present":     of.ob_imbalance >= self._ob_min,
            "not_free_falling":   of.stacked_imb >= self._max_bstack,
            "no_bearish_div":     of.delta_div != 1.0,
            # FIX 3: structure.trend is a TrendDirection enum, not a string.
            # trend != "bearish" compares enum to str → always True (filter never fires).
            "htf_not_bearish":    (not self._htf_align) or structure.trend.value != "bearish",
        }
        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        # Confidence: stronger absorption/OB + meaningful HVN volume.
        hvn_pct = hvn.volume_pct if hvn is not None else 0.0
        conf = min(
            1.0,
            0.35
            + 0.35 * _scale(of.absorption, abs_min, abs_min * 2.0)
            + 0.20 * _scale(of.ob_imbalance, self._ob_min, self._ob_min * 2.0)
            + 0.10 * _scale(hvn_pct, self._min_hvn, self._min_hvn * 2.0),
        )
        return self._make_signal(conditions, confidence=conf)


# ════════════════════════════════════════════════════════════════════════════════
#  2. HVN DIVERGENCE LONG
#
#  Bullish delta divergence AT a High Volume Node.
#  Without the HVN, divergence can persist for many bars before reversing.
#  The HVN gives the structural reason the divergence will resolve bullishly.
# ════════════════════════════════════════════════════════════════════════════════

class HVNDivergenceLong(SignalModule):
    label = "hvn_divergence_long"
    side  = OrderSide.BUY

    def __init__(
        self,
        absorption_max: float = 0.12,
        ob_imb_min: float = 0.03,
        min_hvn_volume_pct: float = 0.02,      # Slightly looser — divergence adds confidence
        max_bearish_stack: float = -5.0,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_max    = absorption_max
        self._ob_min     = ob_imb_min
        self._min_hvn    = min_hvn_volume_pct
        self._max_bstack = max_bearish_stack
        self._htf_align  = require_htf_align

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid:
            return None

        of  = snap.ltf.flow
        hvn = vp.nearest_hvn_below

        conditions = {
            "at_hvn_support":        vp.at_hvn_below,
            "hvn_meaningful":        hvn is not None and hvn.volume_pct >= self._min_hvn,
            "not_at_lvn":            not vp.at_lvn,
            "bullish_divergence":    of.delta_div == -1.0,
            "sellers_not_absorbing": of.absorption >= -self._abs_max,
            "ob_bid_present":        of.ob_imbalance >= self._ob_min,
            "not_free_falling":      of.stacked_imb >= self._max_bstack,
            # FIX 3: enum vs string comparison — use .value
            "htf_allows_long":       (not self._htf_align) or (
                structure.trend.value != "bearish" or structure.structure_break
            ),
        }
        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        hvn_pct = hvn.volume_pct if hvn is not None else 0.0
        # Divergence signal: emphasize divergence + OB + HVN quality.
        conf = min(
            1.0,
            0.35
            + 0.25 * (1.0 if of.delta_div == -1.0 else 0.0)
            + 0.20 * _scale(of.ob_imbalance, self._ob_min, self._ob_min * 2.0)
            + 0.10 * _scale(hvn_pct, self._min_hvn, self._min_hvn * 2.0)
            + 0.10 * _scale(of.absorption, -self._abs_max, self._abs_max),
        )
        return self._make_signal(conditions, confidence=conf)


# ════════════════════════════════════════════════════════════════════════════════
#  3. POC RECLAIM LONG
#
#  Price broke below POC (bearish), now reclaiming it with volume.
#  POC reclaim = shift from bearish to bullish market context.
#  Only fires when price crosses back above POC with absorption.
#
#  This replaces "wall breakout long" with a more precise VP-native concept.
# ════════════════════════════════════════════════════════════════════════════════

class POCReclaimLong(SignalModule):
    label = "poc_reclaim_long"
    side  = OrderSide.BUY

    def __init__(
        self,
        absorption_min: float = 0.08,
        ob_imb_min: float = 0.08,
        imb_min: float = 0.12,
        poc_proximity_bps: float = 20.0,      # how close to POC counts as "at POC"
        poc_absorption_min: float = 0.08,     # POC-specific absorption requirement
        poc_ob_imb_min: float = 0.08,         # POC-specific OB imbalance requirement
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min    = absorption_min
        self._ob_min     = ob_imb_min
        self._imb_min    = imb_min
        self._poc_prox   = poc_proximity_bps
        self._poc_abs_min = poc_absorption_min
        self._poc_ob_min = poc_ob_imb_min
        self._htf_align  = require_htf_align
        # Track previous signed distance to detect true reclaim events.
        # NOTE: POC itself moves as the VP window rolls; we must guard against false
        # "crosses" caused by POC moving rather than price crossing.
        self._prev_poc_price: float | None = None
        self._prev_signed_dist_bps: float | None = None

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid:
            return None
        if vp.poc_price is None:
            return None

        of = snap.ltf.flow

        # Price must be within poc_proximity_bps of POC and do a true reclaim cross.
        poc_close = vp.poc_distance_bps <= self._poc_prox
        signed_dist_bps = (snap.ltf.close_price - vp.poc_price) / vp.poc_price * 10_000.0

        # Guard: require POC stability across evaluations (avoid false crosses).
        max_poc_move_bps = 6.0
        poc_move_bps = (
            abs(vp.poc_price - self._prev_poc_price) / vp.poc_price * 10_000.0
            if self._prev_poc_price is not None else 0.0
        )

        reclaiming = (
            self._prev_signed_dist_bps is not None
            and self._prev_signed_dist_bps <= -1.0
            and signed_dist_bps >= 1.0
            and poc_move_bps <= max_poc_move_bps
        )

        conditions = {
            "near_poc":          poc_close,
            "poc_reclaim":       reclaiming,
            "not_at_lvn":        not vp.at_lvn,
            "absorption_hold":   of.absorption >= self._poc_abs_min,  # Use POC-specific
            "cvd_rising":        snap.ltf.cvd_rising,
            "ob_bid_present":    of.ob_imbalance >= self._poc_ob_min,  # Use POC-specific
            "buy_imbalance":     of.imbalance >= self._imb_min,
            "not_reversing":     of.stacked_imb >= 1,
            "no_bearish_div":    of.delta_div != 1.0,
            "htf_not_bearish":   (not self._htf_align) or structure.trend.value != "bearish",
        }
        # Update state after computing signal.
        self._prev_poc_price = vp.poc_price
        self._prev_signed_dist_bps = signed_dist_bps

        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        conf = min(
            1.0,
            0.40
            + 0.30 * _scale(of.absorption, self._poc_abs_min, self._poc_abs_min * 2.0)
            + 0.20 * _scale(of.imbalance, self._imb_min, self._imb_min * 2.0)
            + 0.10 * _scale(of.ob_imbalance, self._poc_ob_min, self._poc_ob_min * 2.0),
        )
        return self._make_signal(conditions, confidence=conf)


# ════════════════════════════════════════════════════════════════════════════════
#  4. VALUE AREA LOW BOUNCE LONG  ← SECONDARY / GATED
#
#  Price tests Value Area Low (VAL) and bounces.
#  VAL = bottom of the 70% volume zone = institutional reference support.
#  Counter-trend entries from VAL have high accuracy when value area holds.
# ════════════════════════════════════════════════════════════════════════════════

class VALBounceLong(SignalModule):
    label = "val_bounce_long"
    side  = OrderSide.BUY

    def __init__(
        self,
        absorption_min: float = 0.10,
        ob_imb_min: float = 0.06,
        val_proximity_bps: float = 12.0,
        large_dom_min: float = 0.08,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min   = absorption_min
        self._ob_min    = ob_imb_min
        self._val_prox  = val_proximity_bps
        self._ldom_min  = large_dom_min
        self._htf_align = require_htf_align

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid:
            return None
        if vp.val_price is None:
            return None

        of = snap.ltf.flow
        ls = of.large_buy_vol + of.large_sell_vol
        large_dom = (of.large_buy_vol - of.large_sell_vol) / ls if ls > 1e-9 else 0.0

        # Price must be near VAL
        current_price = vp.val_price  # we check proximity via VP snapshot
        val_dist_bps = (
            abs(snap.ltf.close_price - vp.val_price) / vp.val_price * 10_000.0
            if snap.ltf.close_price and vp.val_price else 9999.0
        )

        conditions = {
            "near_val":           val_dist_bps <= self._val_prox,
            "not_at_lvn":         not vp.at_lvn,
            "in_or_below_va":     not vp.above_poc,    # price at or below POC
            "hvn_target_exists":  vp.nearest_hvn_above is not None or vp.vah_price is not None,
            "absorption_start":   of.absorption >= self._abs_min,
            "ob_bid_present":     of.ob_imbalance >= self._ob_min,
            "large_dom":          large_dom >= self._ldom_min,
            "cvd_rising":         snap.ltf.cvd_rising,
            "no_bearish_div":     of.delta_div != 1.0,
            "htf_not_bearish":    (not self._htf_align) or structure.trend.value != "bearish",
        }
        return self._make_signal(conditions)


# ════════════════════════════════════════════════════════════════════════════════
#  5. POC ACCEPTANCE + RETEST LONG  ← BREAKOUT/ACCEPTANCE CORE
#
#  Breakout systems should not enter on the impulse candle.
#  They enter after acceptance (holds outside level) and a retest.
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _AcceptanceState:
    stage: str = "idle"               # idle|accepting|accepted
    accept_count: int = 0
    accepted_level: float | None = None
    accepted_ts_ms: int | None = None


class POCAcceptanceRetestLong(SignalModule):
    label = "poc_acceptance_retest_long"
    side = OrderSide.BUY

    def __init__(
        self,
        poc_band_bps: float = 8.0,
        acceptance_evals: int = 2,
        retest_band_bps: float = 10.0,
        retest_window_evals: int = 6,
        absorption_min: float = 0.06,
        ob_imb_min: float = 0.03,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._poc_band = float(poc_band_bps)
        self._accept_n = int(acceptance_evals)
        self._retest_band = float(retest_band_bps)
        self._retest_window = int(retest_window_evals)
        self._abs_min = float(absorption_min)
        self._ob_min = float(ob_imb_min)
        self._htf_align = bool(require_htf_align)
        self._state = _AcceptanceState()
        self._evals_since_accept = 0
        self._prev_signed_dist_bps: float | None = None

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid or vp.poc_price is None:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
            self._prev_signed_dist_bps = None
            return None

        px = float(snap.ltf.close_price)
        of = snap.ltf.flow
        signed = (px - vp.poc_price) / vp.poc_price * 10_000.0
        broke_up = (self._prev_signed_dist_bps is not None) and (
            self._prev_signed_dist_bps <= self._poc_band and signed > self._poc_band
        )
        self._prev_signed_dist_bps = signed

        # Stage 1: detect breakout and start acceptance counter
        if self._state.stage == "idle":
            if broke_up and not vp.at_lvn:
                self._state.stage = "accepting"
                self._state.accept_count = 1
                self._state.accepted_level = float(vp.poc_price)
            return None

        # Stage 2: acceptance = hold above POC band for N evals
        if self._state.stage == "accepting":
            if signed > self._poc_band:
                self._state.accept_count += 1
                if self._state.accept_count >= self._accept_n:
                    self._state.stage = "accepted"
                    self._state.accepted_ts_ms = int(snap.ltf.ts_ms)
                    self._evals_since_accept = 0
            else:
                self._state = _AcceptanceState()  # failed acceptance
            return None

        # Stage 3: retest within band, but still above (no failure back below)
        self._evals_since_accept += 1
        if self._evals_since_accept > self._retest_window:
            self._state = _AcceptanceState()
            return None

        near_poc = abs(signed) <= self._retest_band and signed >= -1.0
        conditions = {
            "accepted_outside_poc": True,
            "retest_near_poc": near_poc,
            "not_at_lvn": not vp.at_lvn,
            # Context must be bullish: blocks longs when value is accepting lower.
            "above_poc_context": signed >= 0.0,
            "cvd_rising": snap.ltf.cvd_rising,
            "absorption_ok": of.absorption >= self._abs_min,
            "ob_ok": of.ob_imbalance >= self._ob_min,
            # Trend-following: in bull runs this blocks countertrend longs from firing in chop.
            "htf_bullish_or_bos": (not self._htf_align) or (
                structure.trend.value == "bullish"
                or (structure.structure_break and structure.break_type == "high")
            ),
        }
        sig = self._make_signal(conditions)
        if sig:
            # Reset after firing to avoid duplicate entries on next eval.
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
        return sig


class VAHAcceptanceLong(SignalModule):
    label = "vah_acceptance_long"
    side = OrderSide.BUY

    def __init__(
        self,
        va_band_bps: float = 10.0,
        acceptance_evals: int = 2,
        retest_band_bps: float = 12.0,
        retest_window_evals: int = 6,
        absorption_min: float = 0.05,
        ob_imb_min: float = 0.03,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._va_band = float(va_band_bps)
        self._accept_n = int(acceptance_evals)
        self._retest_band = float(retest_band_bps)
        self._retest_window = int(retest_window_evals)
        self._abs_min = float(absorption_min)
        self._ob_min = float(ob_imb_min)
        self._htf_align = bool(require_htf_align)
        self._state = _AcceptanceState()
        self._evals_since_accept = 0
        self._prev_dist_bps: float | None = None

    def evaluate(
        self,
        snap: "MultiTFSnapshot",
        structure: "MarketStructureSnapshot",
        session: "SessionState",
        vp: Optional["VolumeProfileSnapshot"] = None,
    ) -> EntrySignal | None:
        if not session.active or vp is None or not vp.is_valid or vp.vah_price is None:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
            self._prev_dist_bps = None
            return None

        px = float(snap.ltf.close_price)
        of = snap.ltf.flow
        dist = (px - vp.vah_price) / vp.vah_price * 10_000.0  # positive above VAH
        broke_up = (self._prev_dist_bps is not None) and (self._prev_dist_bps <= self._va_band and dist > self._va_band)
        self._prev_dist_bps = dist

        if self._state.stage == "idle":
            if broke_up and not vp.at_lvn:
                self._state.stage = "accepting"
                self._state.accept_count = 1
                self._state.accepted_level = float(vp.vah_price)
            return None

        if self._state.stage == "accepting":
            if dist > self._va_band:
                self._state.accept_count += 1
                if self._state.accept_count >= self._accept_n:
                    self._state.stage = "accepted"
                    self._state.accepted_ts_ms = int(snap.ltf.ts_ms)
                    self._evals_since_accept = 0
            else:
                self._state = _AcceptanceState()
            return None

        self._evals_since_accept += 1
        if self._evals_since_accept > self._retest_window:
            self._state = _AcceptanceState()
            return None

        near_vah = abs(dist) <= self._retest_band and dist >= -1.0
        conditions = {
            "accepted_outside_vah": True,
            "retest_near_vah": near_vah,
            "not_at_lvn": not vp.at_lvn,
            "above_poc_context": vp.above_poc,
            "cvd_rising": snap.ltf.cvd_rising,
            "absorption_ok": of.absorption >= self._abs_min,
            "ob_ok": of.ob_imbalance >= self._ob_min,
            "htf_bullish_or_bos": (not self._htf_align) or (
                structure.trend.value == "bullish"
                or (structure.structure_break and structure.break_type == "high")
            ),
        }
        sig = self._make_signal(conditions)
        if sig:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
        return sig