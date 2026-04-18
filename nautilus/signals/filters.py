"""
signals/filters.py — Pre-signal noise gates.

Applied BEFORE any signal module runs. If any filter fails, evaluation short-circuits.
These are NOT signal conditions — they define whether the MARKET is tradeable at all.

Filters:
  1. VolumeActivityFilter   — is there enough real volume? (rejects dead markets)
  2. WaveQualityFilter      — is the recent price action a real move or just noise?
  3. VolumeProfileReadinessFilter — is the volume profile valid and showing structure?
  4. SessionQualityFilter   — are we in a high-quality trading window?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nautilus.features.multi_tf import MultiTFSnapshot
    from nautilus.features.volume_profile import VolumeProfileSnapshot
    from nautilus.sessions.filter import SessionState


@dataclass(slots=True, frozen=True)
class FilterResult:
    passed: bool
    failed_filters: list[str] = field(default_factory=list)

    @staticmethod
    def ok() -> "FilterResult":
        return FilterResult(True, [])

    @staticmethod
    def fail(*reasons: str) -> "FilterResult":
        return FilterResult(False, list(reasons))


# ── Individual filters ─────────────────────────────────────────────────────────

class VolumeActivityFilter:
    """
    Reject setups where total volume is below a minimum threshold.
    Prevents trading dead-market noise where spreads are wide and moves are fake.

    min_volume_ratio: current bar volume / rolling mean volume.
    A ratio of 0.5 means the current bar must have at least 50% of average volume.
    """

    def __init__(self, min_volume_ratio: float = 0.4) -> None:
        self._min_ratio = min_volume_ratio

    def check(self, snap: "MultiTFSnapshot") -> FilterResult:
        of = snap.ltf.flow
        if of.total_vol <= 1e-9:
            return FilterResult.fail("zero_volume")

        # Use the ratio of current vol to average vol if available
        # Fallback: just require non-zero volume
        if hasattr(of, "vol_ratio") and of.vol_ratio is not None:
            if of.vol_ratio < self._min_ratio:
                return FilterResult.fail(f"low_volume_ratio:{of.vol_ratio:.2f}<{self._min_ratio}")

        return FilterResult.ok()


class WaveQualityFilter:
    """
    Reject setups where there is no real directional wave present.

    A "wave" requires:
    - CVD has moved at least min_cvd_move from its recent low/high
    - The price range over the lookback is at least min_range_bps wide
    - We are NOT in the middle of a completed wave (stacked_imb too extreme)

    This prevents trading:
    - Micro-chop (range too small)
    - Exhausted moves (stacked_imb too high — wave already ran)
    - Random delta noise
    """

    def __init__(
        self,
        min_range_bps: float = 8.0,       # minimum price range to justify a trade
        max_stacked_imb: float = 6.0,      # if stacked this high, wave already ran
        min_cvd_slope: float = 0.05,       # minimum CVD momentum (normalized)
    ) -> None:
        self._min_range = min_range_bps
        self._max_stack = max_stacked_imb
        self._min_cvd   = min_cvd_slope

    def check(self, snap: "MultiTFSnapshot") -> FilterResult:
        of  = snap.ltf.flow
        ltf = snap.ltf

        # Block exhausted waves
        if abs(of.stacked_imb) >= self._max_stack:
            return FilterResult.fail(f"exhausted_wave:stacked={of.stacked_imb:.1f}")

        # Block tiny ranges (chop / noise)
        if hasattr(ltf, "range_bps") and ltf.range_bps is not None:
            if ltf.range_bps < self._min_range:
                return FilterResult.fail(f"range_too_small:{ltf.range_bps:.1f}bps<{self._min_range}")

        return FilterResult.ok()


class VolumeProfileReadinessFilter:
    """
    Gate on volume profile quality and structure.

    Requires:
    - VP is valid (enough buckets, sufficient total volume)
    - For LONG entries: price is at or below POC with HVNs available below (support context)
    - For SHORT entries: price is at or above POC with HVNs available above (resistance context)
    - Never enter at an LVN (low volume node — no structural support, price moves through quickly)

    VP provides institutional reference levels (POC, VAH, VAL, HVN/LVN) that are
    self-fulfilling because they appear on professional charts worldwide.
    """

    def __init__(
        self,
        require_hvn_structure: bool = True,  # must see HVNs in the entry direction
        reject_at_lvn: bool = True,           # never enter at an LVN
        require_valid: bool = True,           # VP must be warm and valid
    ) -> None:
        self._require_hvn = require_hvn_structure
        self._reject_lvn = reject_at_lvn
        self._require_valid = require_valid

    def check(
        self,
        vp: Optional["VolumeProfileSnapshot"],
        is_long: bool,
    ) -> FilterResult:
        if vp is None:
            return FilterResult.fail("no_volume_profile")

        if self._require_valid and not vp.is_valid:
            return FilterResult.fail("vp_not_valid")

        # Reject if price is at an LVN — no structural support
        if self._reject_lvn and vp.at_lvn:
            return FilterResult.fail("at_lvn")

        # For LONG: require HVNs below price (support context) or at least POC below
        # For SHORT: require HVNs above price (resistance context) or at least POC above
        if self._require_hvn:
            if is_long:
                # hvn_below is a list — falsy when empty (no HVN support nodes found)
                if not vp.hvn_below and vp.poc_price is None:
                    return FilterResult.fail("no_hvn_support")
                # FIX 4: removed poc_too_weak check — it compared vp.poc_price (a
                # price level, e.g. 50 000) to vp.total_volume / 10 (a volume
                # quantity, e.g. 80 BTC).  Price > volume for any real asset, so
                # the condition was always False and never fired.  HVN list check
                # above already gates on structural quality.
            else:
                if not vp.hvn_above and vp.poc_price is None:
                    return FilterResult.fail("no_hvn_resistance")

        return FilterResult.ok()


class SessionQualityFilter:
    """
    Only trade during named high-quality sessions.
    Configurable — pass allowed_sessions=None to trade all sessions.
    """

    def __init__(
        self,
        allowed_sessions: Optional[list[str]] = None,  # None = all sessions allowed
        require_active: bool = True,
    ) -> None:
        self._allowed = set(allowed_sessions) if allowed_sessions else None
        self._require_active = require_active

    def check(self, session: "SessionState") -> FilterResult:
        if self._require_active and not session.active:
            return FilterResult.fail("session_not_active")

        if self._allowed is not None:
            name = getattr(session, "session_name", None) or ""
            if name not in self._allowed:
                return FilterResult.fail(f"session_not_allowed:{name}")

        return FilterResult.ok()


# ── Composite filter stack ─────────────────────────────────────────────────────

class NoiseFilterStack:
    """
    Runs all filters in order. Returns first failure, or ok if all pass.

    Usage in strategy:
        stack = NoiseFilterStack(...)
        result = stack.check(snap, vp, session, is_long=True)
        if not result.passed:
            return None  # blocked by noise filter
    """

    def __init__(
        self,
        volume: Optional[VolumeActivityFilter] = None,
        wave: Optional[WaveQualityFilter] = None,
        vp_filter: Optional[VolumeProfileReadinessFilter] = None,
        session: Optional[SessionQualityFilter] = None,
    ) -> None:
        self._volume   = volume   or VolumeActivityFilter()
        self._wave     = wave     or WaveQualityFilter()
        self._vp       = vp_filter or VolumeProfileReadinessFilter()
        self._session  = session  or SessionQualityFilter()

    def check(
        self,
        snap: "MultiTFSnapshot",
        vp: Optional["VolumeProfileSnapshot"],
        session: "SessionState",
        is_long: bool,
    ) -> FilterResult:
        for result in (
            self._volume.check(snap),
            self._wave.check(snap),
            self._vp.check(vp, is_long=is_long),
            self._session.check(session),
        ):
            if not result.passed:
                return result
        return FilterResult.ok()

    @classmethod
    def default(cls) -> "NoiseFilterStack":
        return cls(
            # Production defaults: block low-quality market states before signals.
            volume=VolumeActivityFilter(min_volume_ratio=0.4),
            wave=WaveQualityFilter(min_range_bps=8.0, max_stacked_imb=6.0),
            vp_filter=VolumeProfileReadinessFilter(
                require_hvn_structure=True,
                reject_at_lvn=True,
                require_valid=True,
            ),
            session=SessionQualityFilter(require_active=True),
        )
