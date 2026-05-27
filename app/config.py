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
    odds_api_key: str | None = None
    odds_api_base_url: str = "https://api.odds-api.io/v3"
    odds_bookmakers: str = "DraftKings,FanDuel,BetMGM,Caesars"
    odds_default_sports: str = "basketball,baseball,american-football,ice-hockey"
    sgo_api_key: str | None = None
    sgo_base_url: str = "https://api.sportsgameodds.com/v2"
    sgo_enabled: bool = False
    mlb_stats_enabled: bool = True
    weather_api_key: str | None = None
    weather_api_base_url: str = "https://api.weatherapi.com/v1"
    pybaseball_enabled: bool = True
    allow_live_pybaseball_requests: bool = False
    statcast_cache_last_n_days: int = 14
    statcast_cache_player_ids: str = ""
    max_live_statcast_days: int = 7
    max_live_statcast_rows: int = 500
    mlb_discord_min_score: float = 80.0
    mlb_edge_default_game_date: str | None = None

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

    # --- personal P&L tracker ---
    # Your Polymarket wallet (public on-chain address; no key required — we
    # only read positions/trades from the public data API).
    my_polymarket_wallet: str | None = None
    # Comma-separated to support multiple wallets / sub-accounts.
    my_polymarket_wallets: str = ""

    # Your Kalshi account. The v2 API uses RSA-PSS signed headers; supply the
    # API key id plus a PEM-encoded RSA private key (file path *or* literal
    # PEM string). Auth is skipped if either is missing.
    kalshi_user_api_key_id: str | None = None
    kalshi_user_private_key_path: str | None = None
    kalshi_user_private_key_pem: str | None = None

    # If True, wallet sync uses local mocks so the tracker is exercisable
    # without API keys. Forced on automatically when no creds are present.
    pnl_use_mock_wallet: bool = False

    # Floors for the smart-alert engine.
    pnl_alert_overexposure_pct: float = 0.20       # 20% of portfolio on one game
    pnl_alert_overexposure_sport_pct: float = 0.60 # 60% of portfolio on one sport
    pnl_alert_negative_edge_threshold: float = -0.02  # |fair - market| in cents
    pnl_alert_strong_clv_points: float = 0.04      # 4 cents of CLV
    pnl_alert_bad_entry_points: float = 0.04       # 4 cents worse than callout

    # --- nested ---
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)
    risk: RiskLimits = Field(default_factory=RiskLimits)

    def has_falcon_credentials(self) -> bool:
        return bool(self.falcon_api_key)

    def has_polymarket_credentials(self) -> bool:
        return bool(self.polymarket_api_key)

    def has_kalshi_credentials(self) -> bool:
        return bool(self.kalshi_api_key)

    def has_odds_api_credentials(self) -> bool:
        return bool(self.odds_api_key)

    def has_weather_api_credentials(self) -> bool:
        return bool(self.weather_api_key)

    def has_polymarket_wallet_addresses(self) -> bool:
        return bool(self.my_polymarket_wallet) or bool(self.my_polymarket_wallets.strip())

    def has_kalshi_user_credentials(self) -> bool:
        return bool(self.kalshi_user_api_key_id) and bool(
            self.kalshi_user_private_key_path or self.kalshi_user_private_key_pem
        )

    def polymarket_wallet_list(self) -> list[str]:
        addresses: list[str] = []
        if self.my_polymarket_wallet:
            addresses.append(self.my_polymarket_wallet.strip())
        for chunk in self.my_polymarket_wallets.split(","):
            chunk = chunk.strip()
            if chunk and chunk not in addresses:
                addresses.append(chunk)
        return addresses


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — import this everywhere instead of re-instantiating."""
    return Settings()
