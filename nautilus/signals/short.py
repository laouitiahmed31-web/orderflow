"""
signals/short.py — Volume-profile-anchored short entry signals.

Mirror of long.py. All short entries require price AT an HVN above (resistance).
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


class HVNAbsorptionShort(SignalModule):
    label = "hvn_absorption_short"
    side  = OrderSide.SELL

    def __init__(
        self,
        absorption_min: float = 0.08,
        ob_imb_min: float = 0.05,
        min_hvn_volume_pct: float = 0.03,
        max_bullish_stack: float = 3.0,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min    = absorption_min
        self._ob_min     = ob_imb_min
        self._min_hvn    = min_hvn_volume_pct
        self._max_bstack = max_bullish_stack
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
        hvn = vp.nearest_hvn_above

        abs_min = self._abs_min if vp.below_poc else self._abs_min * 1.5

        conditions = {
            "at_hvn_resistance":   vp.at_hvn_above,
            "hvn_meaningful":      hvn is not None and hvn.volume_pct >= self._min_hvn,
            "not_at_lvn":          not vp.at_lvn,
            "hvn_target_exists":   vp.nearest_hvn_below is not None,
            "absorption_start":    of.absorption <= -abs_min,
            "directional_signal":  (not snap.ltf.cvd_rising) or of.delta_div == 1.0,
            "ob_ask_present":      of.ob_imbalance <= -self._ob_min,
            "not_parabolic":       of.stacked_imb <= self._max_bstack,
            "no_bullish_div":      of.delta_div != -1.0,
            "htf_not_bullish":     (not self._htf_align) or structure.trend.value != "bullish",  # FIX 3: enum vs string
        }
        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        hvn_pct = hvn.volume_pct if hvn is not None else 0.0
        # For shorts, absorption and OB imbalance are negative; use magnitude.
        conf = min(
            1.0,
            0.35
            + 0.35 * _scale(-of.absorption, abs_min, abs_min * 2.0)
            + 0.20 * _scale(-of.ob_imbalance, self._ob_min, self._ob_min * 2.0)
            + 0.10 * _scale(hvn_pct, self._min_hvn, self._min_hvn * 2.0),
        )
        return self._make_signal(conditions, confidence=conf)


class HVNDivergenceShort(SignalModule):
    label = "hvn_divergence_short"
    side  = OrderSide.SELL

    def __init__(
        self,
        absorption_max: float = 0.12,
        ob_imb_min: float = 0.03,
        min_hvn_volume_pct: float = 0.02,
        max_bullish_stack: float = 5.0,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_max    = absorption_max
        self._ob_min     = ob_imb_min
        self._min_hvn    = min_hvn_volume_pct
        self._max_bstack = max_bullish_stack
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
        hvn = vp.nearest_hvn_above

        conditions = {
            "at_hvn_resistance":     vp.at_hvn_above,
            "hvn_meaningful":        hvn is not None and hvn.volume_pct >= self._min_hvn,
            "not_at_lvn":            not vp.at_lvn,
            "bearish_divergence":    of.delta_div == 1.0,
            "buyers_not_absorbing":  of.absorption <= self._abs_max,
            "ob_ask_present":        of.ob_imbalance <= -self._ob_min,
            "not_parabolic":         of.stacked_imb <= self._max_bstack,
            "htf_allows_short":      (not self._htf_align) or (
                structure.trend.value != "bullish" or structure.structure_break
            ),
        }
        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        hvn_pct = hvn.volume_pct if hvn is not None else 0.0
        conf = min(
            1.0,
            0.35
            + 0.25 * (1.0 if of.delta_div == 1.0 else 0.0)
            + 0.20 * _scale(-of.ob_imbalance, self._ob_min, self._ob_min * 2.0)
            + 0.10 * _scale(hvn_pct, self._min_hvn, self._min_hvn * 2.0)
            + 0.10 * _scale(-of.absorption, -self._abs_max, self._abs_max),
        )
        return self._make_signal(conditions, confidence=conf)


class POCRejectionShort(SignalModule):
    """
    Price attempts to reclaim POC from below but gets rejected.
    POC rejection = market confirming bearish context.
    """

    label = "poc_rejection_short"
    side  = OrderSide.SELL

    def __init__(
        self,
        absorption_min: float = 0.08,
        ob_imb_min: float = 0.08,
        imb_min: float = 0.12,
        poc_proximity_bps: float = 20.0,
        poc_absorption_min: float = 0.08,     # POC-specific absorption requirement
        poc_ob_imb_min: float = 0.08,         # POC-specific OB imbalance requirement
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min   = absorption_min
        self._ob_min    = ob_imb_min
        self._imb_min   = imb_min
        self._poc_prox  = poc_proximity_bps
        self._poc_abs_min = poc_absorption_min
        self._poc_ob_min = poc_ob_imb_min
        self._htf_align = require_htf_align
        # Track previous signed distance to detect true rejection events, with POC stability guard.
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

        poc_close = vp.poc_distance_bps <= self._poc_prox
        signed_dist_bps = (snap.ltf.close_price - vp.poc_price) / vp.poc_price * 10_000.0
        max_poc_move_bps = 6.0
        poc_move_bps = (
            abs(vp.poc_price - self._prev_poc_price) / vp.poc_price * 10_000.0
            if self._prev_poc_price is not None else 0.0
        )
        rejected = (
            self._prev_signed_dist_bps is not None
            and self._prev_signed_dist_bps >= 1.0
            and signed_dist_bps <= -1.0
            and poc_move_bps <= max_poc_move_bps
        )

        conditions = {
            "near_poc":           poc_close,
            "poc_rejected":       rejected,
            "not_at_lvn":         not vp.at_lvn,
            "absorption_hold":    of.absorption <= -self._poc_abs_min,  # Use POC-specific
            "cvd_falling":        not snap.ltf.cvd_rising,
            "ob_ask_present":     of.ob_imbalance <= -self._poc_ob_min,  # Use POC-specific
            "sell_imbalance":     of.imbalance <= -self._imb_min,
            "not_reversing":      of.stacked_imb <= -1,
            "no_bullish_div":     of.delta_div != -1.0,
            "htf_not_bullish":    (not self._htf_align) or structure.trend.value != "bullish",  # FIX 3: enum vs string
        }
        self._prev_poc_price = vp.poc_price
        self._prev_signed_dist_bps = signed_dist_bps

        def _scale(x: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 1.0
            return max(0.0, min(1.0, (lo - x) / (lo - hi))) if hi < lo else max(0.0, min(1.0, (x - lo) / (hi - lo)))

        # For shorts, absorption and imbalance are negative; use magnitude below -min.
        abs_mag = -of.absorption
        imb_mag = -of.imbalance
        ob_mag = -of.ob_imbalance
        conf = min(
            1.0,
            0.40
            + 0.30 * _scale(abs_mag, self._poc_abs_min, self._poc_abs_min * 2.0)
            + 0.20 * _scale(imb_mag, self._imb_min, self._imb_min * 2.0)
            + 0.10 * _scale(ob_mag, self._poc_ob_min, self._poc_ob_min * 2.0),
        )
        return self._make_signal(conditions, confidence=conf)


class VAHRejectionShort(SignalModule):
    """
    Price tests Value Area High (VAH) and gets rejected.
    VAH = top of the 70% volume zone = institutional reference resistance.
    """

    label = "vah_rejection_short"
    side  = OrderSide.SELL

    def __init__(
        self,
        absorption_min: float = 0.10,
        ob_imb_min: float = 0.06,
        vah_proximity_bps: float = 12.0,
        large_dom_min: float = 0.08,
        require_htf_align: bool = True,
        **_,
    ) -> None:
        self._abs_min   = absorption_min
        self._ob_min    = ob_imb_min
        self._vah_prox  = vah_proximity_bps
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
        if vp.vah_price is None:
            return None

        of = snap.ltf.flow
        ls = of.large_buy_vol + of.large_sell_vol
        large_dom = (of.large_buy_vol - of.large_sell_vol) / ls if ls > 1e-9 else 0.0

        vah_dist_bps = (
            abs(snap.ltf.close_price - vp.vah_price) / vp.vah_price * 10_000.0
            if snap.ltf.close_price and vp.vah_price else 9999.0
        )

        conditions = {
            "near_vah":           vah_dist_bps <= self._vah_prox,
            "not_at_lvn":         not vp.at_lvn,
            "above_or_at_va":     vp.above_poc,
            "hvn_target_exists":  vp.nearest_hvn_below is not None or vp.val_price is not None,
            "absorption_start":   of.absorption <= -self._abs_min,
            "ob_ask_present":     of.ob_imbalance <= -self._ob_min,
            "large_dom_bearish":  large_dom <= -self._ldom_min,
            "cvd_falling":        not snap.ltf.cvd_rising,
            "no_bullish_div":     of.delta_div != -1.0,
            "htf_not_bullish":    (not self._htf_align) or structure.trend.value != "bullish",  # FIX 3: enum vs string
        }
        return self._make_signal(conditions)


# ════════════════════════════════════════════════════════════════════════════════
#  Breakout/acceptance modules (short side)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _AcceptanceState:
    stage: str = "idle"               # idle|accepting|accepted
    accept_count: int = 0
    accepted_level: float | None = None
    accepted_ts_ms: int | None = None


class POCAcceptanceRetestShort(SignalModule):
    label = "poc_acceptance_retest_short"
    side = OrderSide.SELL

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
        broke_down = (self._prev_signed_dist_bps is not None) and (
            self._prev_signed_dist_bps >= -self._poc_band and signed < -self._poc_band
        )
        self._prev_signed_dist_bps = signed

        if self._state.stage == "idle":
            if broke_down and not vp.at_lvn:
                self._state.stage = "accepting"
                self._state.accept_count = 1
                self._state.accepted_level = float(vp.poc_price)
            return None

        if self._state.stage == "accepting":
            if signed < -self._poc_band:
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

        near_poc = abs(signed) <= self._retest_band and signed <= 1.0
        # Institutional bearish acceptance: shorts should come from value accepting lower,
        # not just a brief dip under POC during an overall bull regime.
        below_val = (vp.val_price is not None) and (px < vp.val_price)
        conditions = {
            "accepted_outside_poc": True,
            "retest_near_poc": near_poc,
            "not_at_lvn": not vp.at_lvn,
            # Context must be bearish: this blocks shorts during bull value acceptance.
            "below_poc_context": signed <= 0.0,
            "below_val_context": below_val,
            "cvd_falling": not snap.ltf.cvd_rising,
            "absorption_ok": of.absorption <= -self._abs_min,
            "ob_ok": of.ob_imbalance <= -self._ob_min,
            # Trend-following: only short when HTF is bearish or breaking down.
            "htf_bearish_or_bos": (not self._htf_align) or (
                structure.trend.value == "bearish"
                or (structure.structure_break and structure.break_type == "low")
            ),
        }
        sig = self._make_signal(conditions)
        if sig:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
        return sig


class VALAcceptanceShort(SignalModule):
    label = "val_acceptance_short"
    side = OrderSide.SELL

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
        if not session.active or vp is None or not vp.is_valid or vp.val_price is None:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
            self._prev_dist_bps = None
            return None

        px = float(snap.ltf.close_price)
        of = snap.ltf.flow
        dist = (px - vp.val_price) / vp.val_price * 10_000.0  # negative below VAL
        broke_down = (self._prev_dist_bps is not None) and (self._prev_dist_bps >= -self._va_band and dist < -self._va_band)
        self._prev_dist_bps = dist

        if self._state.stage == "idle":
            if broke_down and not vp.at_lvn:
                self._state.stage = "accepting"
                self._state.accept_count = 1
                self._state.accepted_level = float(vp.val_price)
            return None

        if self._state.stage == "accepting":
            if dist < -self._va_band:
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

        near_val = abs(dist) <= self._retest_band and dist <= 1.0
        conditions = {
            "accepted_outside_val": True,
            "retest_near_val": near_val,
            "not_at_lvn": not vp.at_lvn,
            "below_poc_context": vp.below_poc,
            # Ensure we're actually accepting below value, not just tagging VAL.
            "below_val_context": dist < 0.0,
            "cvd_falling": not snap.ltf.cvd_rising,
            "absorption_ok": of.absorption <= -self._abs_min,
            "ob_ok": of.ob_imbalance <= -self._ob_min,
            "htf_bearish_or_bos": (not self._htf_align) or (
                structure.trend.value == "bearish"
                or (structure.structure_break and structure.break_type == "low")
            ),
        }
        sig = self._make_signal(conditions)
        if sig:
            self._state = _AcceptanceState()
            self._evals_since_accept = 0
        return sig