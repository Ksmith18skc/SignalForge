"""Application configuration.

All runtime knobs live here. Values are loaded from environment variables (or a
local `.env` file) via pydantic-settings. Nothing is hardcoded as a secret.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CopyMode = Literal["disabled", "alert_only", "paper", "live"]


class ScoringWeights(BaseModel):
    """Weighted components of the signal score. Must sum to 1.0.

    Override via SIGNALFORGE_SCORING__WALLET_QUALITY etc.
    """

    wallet_quality: float = 0.35
    multi_wallet_consensus: float = 0.25
    liquidity: float = 0.15
    entry_timing: float = 0.15
    price_inefficiency: float = 0.10


class RiskLimits(BaseModel):
    """Hard caps used by the risk service. Alerts only — no real trades.

    Override via SIGNALFORGE_RISK__BANKROLL_USD etc.
    """

    max_position_size_pct: float = 0.05   # 5% of bankroll per position
    max_daily_exposure_pct: float = 0.25  # 25% of bankroll deployed per day
    max_per_market_exposure_pct: float = 0.10  # 10% per single market
    bankroll_usd: float = 10_000.0


class Settings(BaseSettings):
    """Top-level settings loaded from environment / .env.

    Every env var is prefixed with `SIGNALFORGE_` so SignalForge config never
    collides with unrelated shell vars (e.g. another project's DATABASE_URL).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SIGNALFORGE_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime ---
    app_name: str = "SignalForge"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    cors_allow_origins: str = ""
    api_url: str = "http://127.0.0.1:8000"
    auto_seed_watchlist: bool = True

    # --- storage ---
    database_url: str = "sqlite:///./signalforge.db"

    # --- trading posture (MVP is alert-only) ---
    default_copy_mode: CopyMode = "alert_only"
    enable_auto_trading: bool = False
    enable_paper_trading: bool = False

    # --- scanner ---
    scan_interval_seconds: int = 60
    signal_score_threshold: float = 50.0  # only persist/alert above this

    # --- provider credentials (optional — missing = MockProvider) ---
    falcon_api_key: str | None = None
    falcon_base_url: str = "https://narrative.agent.heisenberg.so"
    polymarket_api_key: str | None = None
    polymarket_base_url: str = "https://gamma-api.polymarket.com"
    kalshi_api_key: str | None = None
    kalshi_base_url: str = "https://trading-api.kalshi.com/trade-api/v2"

    # --- alert channels ---
    discord_webhook_url: str | None = None
    discord_bot_token: str | None = None
    discord_guild_id: int | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    alert_email_to: str | None = None
    alert_email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    min_discord_score: float = 70.0
    min_trade_size_usd: float = 100.0
    duplicate_window_minutes: int = 30
    high_conviction_score: float = 80.0
    possible_entry_score: float = 85.0

    # --- nested ---
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    risk: RiskLimits = Field(default_factory=RiskLimits)

    def has_falcon_credentials(self) -> bool:
        return bool(self.falcon_api_key)

    def has_polymarket_credentials(self) -> bool:
        return bool(self.polymarket_api_key)

    def has_kalshi_credentials(self) -> bool:
        return bool(self.kalshi_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — import this everywhere instead of re-instantiating."""
    return Settings()
