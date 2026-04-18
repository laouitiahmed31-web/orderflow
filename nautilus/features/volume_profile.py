"""
features/volume_profile.py — Session and rolling volume profile engine.

Replaces LiquidityHeatmap. Key difference:
  Heatmap = traded volume + resting OB volume combined.
  Volume Profile = traded volume only, organized as a price distribution.

Why volume profile is better here:
  - OB resting volume is easy to spoof. Traded volume cannot be faked.
  - VP gives you POC, VAH, VAL — institutional reference levels that actually
    appear on professional charts and are self-fulfilling because everyone sees them.
  - HVNs (high volume nodes) are support/resistance. Price stalls and reverses there.
  - LVNs (low volume nodes) are the opposite — price moves THROUGH them fast.
    Entering at an LVN is entering into air. Bad.

Key output levels:
  POC  — Point of Control: single price with most volume. Magnet.
  VAH  — Value Area High: top of the 70% volume zone. Resistance in uptrend.
  VAL  — Value Area Low: bottom of the 70% volume zone. Support in downtrend.
  HVN  — High Volume Node: local volume peak. Acts like a wall (entry zone).
  LVN  — Low Volume Node: local volume trough. Price moves through fast (avoid entry here).

Strategy use:
  - Enter LONG at HVN below POC (or VAL) with absorption confirming.
  - Enter SHORT at HVN above POC (or VAH) with selling confirming.
  - Stop below/above the HVN (if HVN breaks, volume context is gone).
  - Target the next HVN in the direction of the trade.
  - NEVER enter at an LVN — no structural support, no meaningful level.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class VolumeNode:
    price: float
    volume: float
    volume_pct: float        # share of total profile volume [0, 1]
    distance_bps: float      # distance from current price (always positive)
    node_type: str           # "HVN" | "LVN" | "POC"


@dataclass(slots=True)
class VolumeProfileSnapshot:
    """
    Point-in-time VP output consumed by signals and filters.

    Naming convention mirrors standard VP software (Sierra Chart, Bookmap, etc.)
    so this is directly comparable to what discretionary traders see.
    """
    poc_price: Optional[float] = None          # Point of Control
    vah_price: Optional[float] = None          # Value Area High
    val_price: Optional[float] = None          # Value Area Low
    poc_distance_bps: float = 9999.0           # distance from current price to POC

    # Band proximity (used for acceptance / retest logic)
    at_poc: bool = False                       # within poc_band_bps of POC
    at_vah: bool = False                       # within va_band_bps of VAH
    at_val: bool = False                       # within va_band_bps of VAL

    hvn_above: list[VolumeNode] = field(default_factory=list)   # HVNs above current price (EXCLUDES POC)
    hvn_below: list[VolumeNode] = field(default_factory=list)   # HVNs below current price (EXCLUDES POC)
    lvn_above: list[VolumeNode] = field(default_factory=list)   # LVNs above current price
    lvn_below: list[VolumeNode] = field(default_factory=list)   # LVNs below current price

    # Nearest actionable levels (EXCLUDES POC unless explicitly requested)
    nearest_hvn_above: Optional[VolumeNode] = None
    nearest_hvn_below: Optional[VolumeNode] = None
    nearest_lvn_above: Optional[VolumeNode] = None
    nearest_lvn_below: Optional[VolumeNode] = None

    # POC as its own node (context, not an HVN target)
    poc_node: Optional[VolumeNode] = None

    # Proximity flags (price is AT a meaningful level)
    at_hvn: bool = False                # within proximity_bps of any HVN
    at_hvn_below: bool = False          # at an HVN that is below price (support)
    at_hvn_above: bool = False          # at an HVN that is above price (resistance)
    at_lvn: bool = False                # at an LVN (avoid entry — no structure)
    in_value_area: bool = False         # price between VAL and VAH
    above_poc: bool = False             # bullish context
    below_poc: bool = False             # bearish context

    # Bracket anchors (driven by nearest HVN levels)
    long_stop_price: Optional[float] = None    # just below nearest HVN below
    long_target_price: Optional[float] = None  # nearest HVN above (or VAH)
    short_stop_price: Optional[float] = None   # just above nearest HVN above
    short_target_price: Optional[float] = None # nearest HVN below (or VAL)

    # Travel targets (LVN fast zones) — useful for breakout systems
    long_travel_target_price: Optional[float] = None   # nearest LVN above (if any)
    short_travel_target_price: Optional[float] = None  # nearest LVN below (if any)

    # Profile quality
    total_volume: float = 0.0
    is_valid: bool = False              # False if insufficient data


# ── Engine ─────────────────────────────────────────────────────────────────────

class VolumeProfile:
    """
    Rolling volume profile over a configurable trade window.

    Two modes via `session_mode`:
      False (default): rolling window of last `window_trades` trades.
                       Adapts continuously. Good for intraday.
      True:            session-based — resets at session open.
                       Call reset_session() at the start of each trading session.

    Parameters
    ----------
    bucket_size : float
        Price granularity per bucket. For BTC at $90k, 10.0 = $10 buckets.
        Finer buckets = more precise levels but noisier.
        Coarser buckets = smoother profile but misses micro-levels.
        Rule of thumb: 0.01–0.05% of current price.
    window_trades : int
        Rolling window depth (ignored in session_mode).
    value_area_pct : float
        Fraction of total volume that defines the Value Area (default 0.70 = 70%).
    hvn_percentile : float
        Volume percentile above which a bucket is called an HVN (default 0.75).
        Top 25% by volume = high volume node.
    lvn_percentile : float
        Volume percentile below which a bucket is called an LVN (default 0.25).
        Bottom 25% by volume = low volume node.
    proximity_bps : float
        How close price must be to a level for at_hvn / at_lvn = True.
    min_buckets : int
        Minimum number of populated price buckets before emitting valid snapshots.
    stop_buffer_bps : float
        Extra buffer added/subtracted when computing stop prices from HVN edges.
    session_mode : bool
        If True, profile accumulates for the full session and resets on reset_session().
    """

    def __init__(
        self,
        bucket_size: float = 10.0,
        window_trades: int = 50_000,
        value_area_pct: float = 0.70,
        hvn_percentile: float = 0.75,
        lvn_percentile: float = 0.25,
        proximity_bps: float = 8.0,
        min_buckets: int = 100,
        stop_buffer_bps: float = 20.0,
        poc_band_bps: float = 8.0,
        va_band_bps: float = 10.0,
        session_mode: bool = False,
    ) -> None:
        self._bucket       = bucket_size
        self._window       = window_trades
        self._va_pct       = value_area_pct
        self._hvn_pct      = hvn_percentile
        self._lvn_pct      = lvn_percentile
        self._proximity    = proximity_bps
        self._min_buckets  = min_buckets
        self._stop_buf     = stop_buffer_bps
        self._poc_band_bps = poc_band_bps
        self._va_band_bps  = va_band_bps
        self._session_mode = session_mode

        # Volume distribution: bucket_key → cumulative volume
        self._profile: dict[int, float] = {}

        # Rolling tape for window eviction (not used in session_mode)
        self._tape: deque[tuple[int, float]] = deque(maxlen=window_trades)

        # Total volume in current profile
        self._total_volume: float = 0.0

    # ── Feed ───────────────────────────────────────────────────────────────────

    def add_trade(self, price: float, volume: float) -> None:
        """Add a single trade to the profile."""
        b = self._to_bucket(price)

        if not self._session_mode:
            # Rolling window — evict before append to avoid deque(maxlen) silent pop.
            if len(self._tape) >= self._window:
                old_b, old_vol = self._tape.popleft()
                self._profile[old_b] = max(0.0, self._profile.get(old_b, 0.0) - old_vol)
                self._total_volume = max(0.0, self._total_volume - old_vol)
                if self._profile[old_b] < 1e-12:
                    del self._profile[old_b]
            self._tape.append((b, volume))

        self._profile[b] = self._profile.get(b, 0.0) + volume
        self._total_volume += volume

    def reset_session(self) -> None:
        """
        Reset profile for a new trading session.
        Only meaningful in session_mode=True. Safe to call anytime.
        """
        self._profile.clear()
        self._tape.clear()
        self._total_volume = 0.0

    # ── Snapshot ───────────────────────────────────────────────────────────

    def compute_snapshot(self, current_price: float) -> VolumeProfileSnapshot:
        """Build a VolumeProfileSnapshot relative to current_price."""
        if len(self._profile) < self._min_buckets or self._total_volume < 1e-9:
            return VolumeProfileSnapshot(is_valid=False, total_volume=self._total_volume)

        buckets = sorted(self._profile.items())   # [(bucket_key, volume), ...]
        volumes = [v for _, v in buckets]

        # ── POC ───────────────────────────────────────────────────────────
        poc_b, poc_vol = max(buckets, key=lambda x: x[1])
        poc_price = self._to_price(poc_b)

        # ── Value Area (VAH / VAL) ────────────────────────────────────────
        val_price, vah_price = self._compute_value_area(buckets, poc_b)

        # ── HVN / LVN thresholds ──────────────────────────────────────────
        sorted_vols = sorted(volumes)
        n = len(sorted_vols)
        hvn_thresh = sorted_vols[min(int(n * self._hvn_pct), n - 1)]
        lvn_thresh = sorted_vols[max(int(n * self._lvn_pct) - 1, 0)]

        # ── Classify nodes ────────────────────────────────────────────────
        hvn_above: list[VolumeNode] = []
        hvn_below: list[VolumeNode] = []
        lvn_above: list[VolumeNode] = []
        lvn_below: list[VolumeNode] = []
        poc_node: VolumeNode | None = None

        for b, vol in buckets:
            lp = self._to_price(b)
            dist = abs(lp - current_price) / current_price * 10_000.0
            if dist < 1.0:
                continue   # at current price — ignore

            vol_pct = vol / self._total_volume
            is_poc = (b == poc_b)
            ntype = (
                "POC"
                if is_poc
                else ("HVN" if vol >= hvn_thresh else ("LVN" if vol <= lvn_thresh else None))
            )
            if ntype is None:
                continue

            node = VolumeNode(price=lp, volume=vol, volume_pct=vol_pct,
                              distance_bps=dist, node_type=ntype)

            if ntype == "POC":
                # Keep POC separate: it's context/fair value, not an HVN target.
                poc_node = node
            elif ntype == "HVN":
                if lp > current_price:
                    hvn_above.append(node)
                else:
                    hvn_below.append(node)
            elif ntype == "LVN":
                if lp > current_price:
                    lvn_above.append(node)
                else:
                    lvn_below.append(node)

        # Sort by distance
        hvn_above.sort(key=lambda x: x.distance_bps)
        hvn_below.sort(key=lambda x: x.distance_bps)
        lvn_above.sort(key=lambda x: x.distance_bps)
        lvn_below.sort(key=lambda x: x.distance_bps)

        nearest_hvn_above = hvn_above[0] if hvn_above else None
        nearest_hvn_below = hvn_below[0] if hvn_below else None
        nearest_lvn_above = lvn_above[0] if lvn_above else None
        nearest_lvn_below = lvn_below[0] if lvn_below else None

        # ── Proximity flags ───────────────────────────────────────────────
        at_hvn_below = nearest_hvn_below is not None and nearest_hvn_below.distance_bps <= self._proximity
        at_hvn_above = nearest_hvn_above is not None and nearest_hvn_above.distance_bps <= self._proximity
        at_hvn       = at_hvn_below or at_hvn_above
        at_lvn       = (
            (nearest_lvn_above is not None and nearest_lvn_above.distance_bps <= self._proximity)
            or (nearest_lvn_below is not None and nearest_lvn_below.distance_bps <= self._proximity)
        )

        poc_dist = abs(current_price - poc_price) / current_price * 10_000.0
        in_va    = val_price is not None and vah_price is not None and val_price <= current_price <= vah_price
        above_poc = current_price > poc_price
        below_poc = current_price < poc_price

        at_poc = poc_dist <= self._poc_band_bps
        at_vah = (
            vah_price is not None
            and abs(current_price - vah_price) / current_price * 10_000.0 <= self._va_band_bps
        )
        at_val = (
            val_price is not None
            and abs(current_price - val_price) / current_price * 10_000.0 <= self._va_band_bps
        )

        # ── Bracket prices ────────────────────────────────────────────────
        buf = self._stop_buf / 10_000.0

        # Long: stop below nearest HVN below (or VAL), target nearest HVN above (or VAH)
        long_stop   = nearest_hvn_below.price * (1 - buf) if nearest_hvn_below else (
            val_price * (1 - buf) if val_price else None
        )
        long_target = nearest_hvn_above.price if nearest_hvn_above else vah_price

        # Short: stop above nearest HVN above (or VAH), target nearest HVN below (or VAL)
        short_stop   = nearest_hvn_above.price * (1 + buf) if nearest_hvn_above else (
            vah_price * (1 + buf) if vah_price else None
        )
        short_target = nearest_hvn_below.price if nearest_hvn_below else val_price

        # LVN travel targets for breakout systems (fast zones, often first take-profit)
        long_travel_target = nearest_lvn_above.price if nearest_lvn_above else None
        short_travel_target = nearest_lvn_below.price if nearest_lvn_below else None

        
        return VolumeProfileSnapshot(
            poc_price=poc_price,
            vah_price=vah_price,
            val_price=val_price,
            poc_distance_bps=poc_dist,
            at_poc=at_poc,
            at_vah=at_vah,
            at_val=at_val,
            hvn_above=hvn_above,
            hvn_below=hvn_below,
            lvn_above=lvn_above,
            lvn_below=lvn_below,
            nearest_hvn_above=nearest_hvn_above,
            nearest_hvn_below=nearest_hvn_below,
            nearest_lvn_above=nearest_lvn_above,
            nearest_lvn_below=nearest_lvn_below,
            poc_node=poc_node,
            at_hvn=at_hvn,
            at_hvn_below=at_hvn_below,
            at_hvn_above=at_hvn_above,
            at_lvn=at_lvn,
            in_value_area=in_va,
            above_poc=above_poc,
            below_poc=below_poc,
            long_stop_price=long_stop,
            long_target_price=long_target,
            short_stop_price=short_stop,
            short_target_price=short_target,
            long_travel_target_price=long_travel_target,
            short_travel_target_price=short_travel_target,
            total_volume=self._total_volume,
            is_valid=True,
        )

    # ── Value area computation ─────────────────────────────────────────────────

    def _compute_value_area(
        self,
        buckets: list[tuple[int, float]],
        poc_b: int,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Standard value area algorithm: expand from POC outward until
        value_area_pct of total volume is captured.

        Returns (val_price, vah_price).
        """
        target = self._total_volume * self._va_pct
        vol_dict = dict(buckets)
        all_keys = sorted(vol_dict)

        if poc_b not in vol_dict:
            return None, None

        poc_idx = all_keys.index(poc_b)
        included = {poc_b}
        accumulated = vol_dict[poc_b]

        lo_idx = poc_idx
        hi_idx = poc_idx

        while accumulated < target:
            can_go_lo = lo_idx > 0
            can_go_hi = hi_idx < len(all_keys) - 1

            if not can_go_lo and not can_go_hi:
                break

            next_lo_vol = vol_dict[all_keys[lo_idx - 1]] if can_go_lo else 0.0
            next_hi_vol = vol_dict[all_keys[hi_idx + 1]] if can_go_hi else 0.0

            # Expand toward the higher volume side first (standard VP method)
            if next_hi_vol >= next_lo_vol and can_go_hi:
                hi_idx += 1
                b = all_keys[hi_idx]
                included.add(b)
                accumulated += vol_dict[b]
            elif can_go_lo:
                lo_idx -= 1
                b = all_keys[lo_idx]
                included.add(b)
                accumulated += vol_dict[b]
            else:
                hi_idx += 1
                b = all_keys[hi_idx]
                included.add(b)
                accumulated += vol_dict[b]

        val_price = self._to_price(all_keys[lo_idx])
        vah_price = self._to_price(all_keys[hi_idx])
        return val_price, vah_price

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _to_bucket(self, price: float) -> int:
        return int(price / self._bucket)

    def _to_price(self, bucket_key: int) -> float:
        return bucket_key * self._bucket

    @property
    def is_warm(self) -> bool:
        """True once we have enough populated buckets to be reliable."""
        return len(self._profile) >= self._min_buckets

    @property
    def total_volume(self) -> float:
        return self._total_volume

    @property
    def bucket_count(self) -> int:
        return len(self._profile)
