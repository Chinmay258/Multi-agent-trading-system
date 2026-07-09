"""
core/config.py
--------------
Centralised configuration management for the trading system.

All configuration is driven by environment variables (12-factor app style).
Pydantic BaseSettings handles parsing, validation, and type coercion automatically.
A single get_settings() call anywhere in the codebase returns the same
validated, immutable settings object (cached via lru_cache).

Design decisions:
- Single source of truth: no config scattered across files.
- Fail-fast: missing/invalid env vars raise at startup, not mid-trade.
- Feature flags: PAPER_TRADING, ENABLE_SENTIMENT, etc. control behaviour
  without code changes — safe for ops to toggle via restart.
- Nested settings groups keep the namespace clean and make unit-testing
  individual subsystems easy (patch only the group you need).

Usage:
    from core.config import get_settings
    settings = get_settings()
    print(settings.redis.url)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradingMode(str, Enum):
    """
    PAPER — all orders routed to mock exchange, zero real money risk.
    LIVE  — real orders placed. Requires explicit opt-in and human sign-off.
    """

    PAPER = "paper"
    LIVE = "live"


# ---------------------------------------------------------------------------
# Sub-setting groups (each reads its own prefixed env vars)
# ---------------------------------------------------------------------------


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="trading_db")
    user: str = Field(default="trading_user")
    password: SecretStr = Field(default=SecretStr("trading_pass"))
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo_sql: bool = Field(default=False, description="Log all SQL (dev only)")

    @property
    def url(self) -> str:
        """Async DSN for SQLAlchemy (asyncpg driver)."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Sync DSN for Alembic migrations."""
        pwd = self.password.get_secret_value()
        return f"postgresql+psycopg2://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: SecretStr | None = Field(default=None)
    max_connections: int = Field(default=20)
    socket_timeout: float = Field(default=5.0)
    retry_on_timeout: bool = Field(default=True)

    @property
    def url(self) -> str:
        if self.password:
            pwd = self.password.get_secret_value()
            return f"redis://:{pwd}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class ExchangeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_", extra="ignore")

    name: str = Field(default="binance", description="CCXT exchange id")
    api_key: SecretStr | None = Field(default=None)
    api_secret: SecretStr | None = Field(default=None)
    api_passphrase: SecretStr | None = Field(default=None, description="Required by OKX etc.")
    sandbox: bool = Field(default=True, description="Use exchange testnet")
    rate_limit_ms: int = Field(default=100)
    request_timeout_ms: int = Field(default=10_000)
    ws_ping_interval: int = Field(default=20)
    ws_reconnect_attempts: int = Field(default=5)
    ws_reconnect_delay: float = Field(default=2.0)


class RiskSettings(BaseSettings):
    """
    Hard limits enforced by the Risk Management Agent.
    These are the circuit-breaker values — treat them as sacred.
    Changing them in production requires a deliberate human decision.
    """

    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore")

    max_position_pct: float = Field(
        default=0.02, description="Max single position as fraction of portfolio (2%)"
    )
    max_open_positions: int = Field(default=3)
    min_order_size_usd: float = Field(default=10.0)
    max_order_size_usd: float = Field(default=1000.0)

    max_daily_loss_pct: float = Field(
        default=0.05, description="Halt all trading if daily loss exceeds 5% of portfolio"
    )
    max_total_drawdown_pct: float = Field(
        default=0.15, description="Emergency halt if total drawdown exceeds 15%"
    )

    max_data_staleness_seconds: int = Field(
        default=60, description="Halt if market data is older than this"
    )

    max_agent_crashes: int = Field(default=3)
    crash_window_minutes: int = Field(default=5)

    @field_validator("max_position_pct", "max_daily_loss_pct", "max_total_drawdown_pct")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"Expected fraction 0–1, got {v}")
        return v


class MarketDataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARKET_DATA_", extra="ignore")

    symbols: list[str] = Field(default=["BTC/USDT"])
    ohlcv_timeframes: list[str] = Field(default=["1m", "5m", "1h"])
    ohlcv_history_limit: int = Field(default=500, description="Candles fetched on startup")
    poll_interval_seconds: float = Field(default=5.0, description="REST polling interval")
    orderbook_depth: int = Field(default=20)


class TechnicalAnalysisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", extra="ignore")

    rsi_period: int = Field(default=14)
    rsi_overbought: float = Field(default=70.0)
    rsi_oversold: float = Field(default=30.0)

    macd_fast: int = Field(default=12)
    macd_slow: int = Field(default=26)
    macd_signal: int = Field(default=9)

    bb_period: int = Field(default=20)
    bb_std_dev: float = Field(default=2.0)

    ema_short: int = Field(default=9)
    ema_long: int = Field(default=21)

    min_signal_confidence: float = Field(
        default=0.6, description="Discard signals below this confidence score"
    )
    signal_ttl_seconds: int = Field(
        default=300, description="Signals older than this are ignored by Decision agent"
    )

    # Phase 8 — ML signal engine.
    # Default False as of Phase 5: the walk-forward evaluation showed the ML path
    # does not beat the rule-based baseline out-of-sample (docs/MODEL_CHANGES.md).
    # Set TA_USE_ML_SIGNALS=true to opt back into the XGBoost signal path.
    use_ml_signals: bool = Field(
        default=False,
        description="If True, the TA agent uses an XGBoost model when one "
        "is available, falling back to rules when not.",
    )
    ml_model_path: str | None = Field(
        default=None,
        description="Override path to the model file. If None, ModelRegistry "
        "is queried for the latest model for each symbol/timeframe.",
    )


class SentimentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", extra="ignore")

    enabled: bool = Field(default=False, description="Off for MVP")
    news_api_key: SecretStr | None = Field(default=None)
    fetch_interval_seconds: int = Field(default=300)
    max_articles_per_fetch: int = Field(default=20)
    sentiment_ttl_seconds: int = Field(default=1800)


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONITORING_", extra="ignore")

    heartbeat_interval_seconds: int = Field(default=10)
    heartbeat_timeout_seconds: int = Field(default=30)
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    slack_webhook_url: SecretStr | None = Field(default=None)
    alert_on_agent_crash: bool = Field(default=True)
    alert_on_daily_loss_threshold: bool = Field(default=True)


class MT5Settings(BaseSettings):
    """HTTP server settings for the MetaTrader 5 Expert Advisor WebRequest bridge."""

    model_config = SettingsConfigDict(env_prefix="MT5_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5555)
    listen_port: int = Field(default=5556)
    request_timeout_ms: int = Field(default=5000)
    max_retries: int = Field(default=3)
    enabled: bool = Field(default=False)
    stop_loss_pct: float = Field(default=0.02)
    take_profit_pct: float = Field(default=0.04)

    # Lot-size constraints — set per symbol on the broker side.
    # Tickmill BTCUSD: min=0.01, step=0.01, max=10.0
    # NOTE: minimum order on Tickmill BTCUSD is ~0.01 lots ≈ $700 at current BTC price.
    min_volume: float = Field(default=0.01, description="Minimum lot size for the traded symbol")
    max_volume: float = Field(default=10.0, description="Maximum lot size for the traded symbol")
    volume_step: float = Field(default=0.01, description="Lot size granularity step")

    # Python-side safety-net monitor interval (MT5 native SL/TP fires in ms;
    # this loop is the backup in case the EA misses a fill).
    position_monitor_interval_seconds: int = Field(
        default=5, description="How often the Python SL/TP backup monitor checks open positions"
    )


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object. Aggregates all sub-setting groups.
    Loaded once at startup via get_settings() and cached — treat as immutable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # System identity
    app_name: str = Field(default="TradingSystem")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    debug: bool = Field(default=False)

    # THE most important flag — default to paper, never live by accident
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="paper (default, safe) or live (real money, explicit opt-in)",
    )

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    technical_analysis: TechnicalAnalysisSettings = Field(default_factory=TechnicalAnalysisSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    mt5: MT5Settings = Field(default_factory=MT5Settings)

    # Broker selection: "paper" (default, safe) | "mt5" (real MT5 terminal)
    execution_broker: str = Field(default="paper")

    # Market-data source selection (see data_sources/):
    #   "public" (default, keyless CCXT) | "mt5" (local-only, read-only) | "auto"
    data_source: str = Field(default="public")

    # API server
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)

    # Paper trading starting balance
    paper_initial_balance_usd: float = Field(default=10_000.0)

    @property
    def is_paper_trading(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    def assert_paper_mode(self) -> None:
        """
        Safety guard — call in any code path that must never touch real money.
        Raises RuntimeError if live mode is active.
        """
        if not self.is_paper_trading:
            raise RuntimeError(
                "This operation is restricted to paper trading mode. "
                "Set TRADING_MODE=paper or verify your intent."
            )

    def log_summary(self) -> dict:
        """Non-sensitive summary for structured startup log."""
        return {
            "app_name": self.app_name,
            "environment": self.environment.value,
            "trading_mode": self.trading_mode.value,
            "exchange": self.exchange.name,
            "symbols": self.market_data.symbols,
            "log_level": self.log_level.value,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    lru_cache ensures .env is parsed exactly once at startup.
    In tests, call get_settings.cache_clear() then patch env vars to reload.

    Example:
        from core.config import get_settings
        settings = get_settings()
        if settings.is_paper_trading:
            use_mock_exchange()
    """
    return Settings()
