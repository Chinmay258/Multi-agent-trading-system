"""
data_sources/
-------------
Pluggable market-data layer. The rest of the system asks for a ``DataSource`` via
``get_data_source()`` and never cares which concrete venue is behind it.

Selection (config ``DATA_SOURCE`` / env ``DATA_SOURCE``):
- ``public`` (default) → ``PublicExchangeSource`` (keyless CCXT public data).
- ``mt5``              → ``MT5Source`` (local-only, read-only MetaTrader 5).
- ``auto``            → ``MT5Source`` if the MetaTrader5 package is importable on
                         this host, otherwise ``PublicExchangeSource``.

The default is ``public`` so a fresh clone with no secrets and no MT5 terminal works
out of the box — which is the whole point of the keyless demo.
"""

from __future__ import annotations

from core.config import Settings, get_settings
from core.logging import get_logger
from data_sources.base import DataSource, DataSourceCapabilities
from data_sources.public_exchange import PublicExchangeSource

logger = get_logger("data_sources")

__all__ = [
    "DataSource",
    "DataSourceCapabilities",
    "PublicExchangeSource",
    "get_data_source",
]


def get_data_source(settings: Settings | None = None) -> DataSource:
    """
    Resolve and return the configured market-data source.

    Defaults to the keyless ``PublicExchangeSource``. ``MT5Source`` is imported
    lazily so hosts without the MetaTrader5 package are never affected.
    """
    settings = settings or get_settings()
    choice = (getattr(settings, "data_source", "public") or "public").strip().lower()

    if choice == "public":
        logger.info("data_source_selected", source="public", keyless=True)
        return PublicExchangeSource()

    if choice in ("mt5", "auto"):
        # Import lazily: keep MetaTrader5 out of keyless/Linux environments.
        from data_sources.mt5_source import MT5Source

        if choice == "mt5":
            logger.info("data_source_selected", source="mt5", read_only=True)
            return MT5Source()

        # auto: prefer MT5 only if its package is present, else fall back.
        if MT5Source.is_available():
            logger.info("data_source_selected", source="mt5", mode="auto", read_only=True)
            return MT5Source()
        logger.info("data_source_selected", source="public", mode="auto", keyless=True)
        return PublicExchangeSource()

    logger.warning("data_source_unknown_falling_back", requested=choice, fallback="public")
    return PublicExchangeSource()
