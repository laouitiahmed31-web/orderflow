"""
Signal registry for loading and managing signal modules by configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nautilus_trader.model.enums import OrderSide

from nautilus.signals.base import SignalModule

if TYPE_CHECKING:
    from nautilus.config.schema import SignalsConfig


class SignalRegistry:
    """
    Registry for registered signal modules.

    Loads signal modules from a configuration dict and exposes them
    as a list for evaluation during strategy evaluation cycles.
    """

    def __init__(self, modules: list[SignalModule], require_all: bool = False) -> None:
        """
        Parameters
        ----------
        modules : list[SignalModule]
            Instantiated signal modules ready for evaluation.
        require_all : bool
            If False (default), stop after the first module that fires on each side.
            If True, evaluate every module and return all that pass.
        """
        self._modules = modules
        self._require_all = require_all

    @property
    def modules(self) -> list[SignalModule]:
        """Return all registered modules."""
        return self._modules

    def evaluate_long(self, snapshot, structure, session, vp=None):
        """
        Evaluate all LONG signal modules.

        Returns all EntrySignal objects that pass conditions.
        """
        signals: list = []
        for module in self._modules:
            if module.side != OrderSide.BUY:
                continue
            result = module.evaluate(snapshot, structure, session, vp)
            if result:
                signals.append(result)
        return signals

    def evaluate_short(self, snapshot, structure, session, vp=None):
        """
        Evaluate all SHORT signal modules.

        Returns all EntrySignal objects that pass conditions.
        """
        signals: list = []
        for module in self._modules:
            if module.side != OrderSide.SELL:
                continue
            result = module.evaluate(snapshot, structure, session, vp)
            if result:
                signals.append(result)
        return signals

    @classmethod
    def from_config(cls, config: SignalsConfig) -> SignalRegistry:
        """
        Load signal modules from a SignalsConfig.

        Parameters
        ----------
        config : SignalsConfig
            Configuration with ``long`` / ``short`` module name lists.

        Returns
        -------
        SignalRegistry
            Registry with loaded modules.
        """
        from nautilus.signals.long import (
            HVNAbsorptionLong,
            HVNDivergenceLong,
            POCAcceptanceRetestLong,
            POCReclaimLong,
            VAHAcceptanceLong,
            VALBounceLong,
        )
        from nautilus.signals.short import (
            HVNAbsorptionShort,
            HVNDivergenceShort,
            POCAcceptanceRetestShort,
            POCRejectionShort,
            VALAcceptanceShort,
            VAHRejectionShort,
        )

        module_map = {
            "hvn_absorption_long": HVNAbsorptionLong,
            "hvn_divergence_long": HVNDivergenceLong,
            "poc_acceptance_retest_long": POCAcceptanceRetestLong,
            "poc_reclaim_long": POCReclaimLong,
            "vah_acceptance_long": VAHAcceptanceLong,
            "val_bounce_long": VALBounceLong,
            "hvn_absorption_short": HVNAbsorptionShort,
            "hvn_divergence_short": HVNDivergenceShort,
            "poc_acceptance_retest_short": POCAcceptanceRetestShort,
            "poc_rejection_short": POCRejectionShort,
            "val_acceptance_short": VALAcceptanceShort,
            "vah_rejection_short": VAHRejectionShort,
        }

        kwargs = getattr(config, "module_kwargs", None) or {}

        def load_module(label: str) -> SignalModule:
            if label not in module_map:
                raise ValueError(f"Unknown signal: {label!r}. Available: {sorted(module_map)}")
            return module_map[label](**kwargs)

        long_modules = [load_module(l) for l in getattr(config, "long", [])]
        short_modules = [load_module(l) for l in getattr(config, "short", [])]
        require_all = bool(getattr(config, "require_all", False))

        return cls(long_modules + short_modules, require_all)

    def __repr__(self) -> str:
        return f"SignalRegistry(modules={len(self._modules)}, require_all={self._require_all})"
