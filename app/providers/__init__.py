"""Data provider abstractions for SignalForge."""

from app.providers.base import BaseProvider, ProviderSource
from app.providers.falcon import FalconProvider
from app.providers.kalshi import KalshiProvider
from app.providers.mlb_stats_api import MlbStatsApiProvider
from app.providers.mock import MockProvider
from app.providers.polymarket import PolymarketProvider
from app.providers.pybaseball_provider import PyBaseballProvider
from app.providers.weather_api import WeatherApiProvider

__all__ = [
    "BaseProvider",
    "ProviderSource",
    "FalconProvider",
    "KalshiProvider",
    "MockProvider",
    "PolymarketProvider",
]
