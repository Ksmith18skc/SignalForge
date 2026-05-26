"""Discord slash-command bot for SignalForge.

Run with:
    python -m scripts.discord_bot

Required env:
    SIGNALFORGE_DISCORD_BOT_TOKEN=...
    SIGNALFORGE_API_URL=https://signalforge-0yap.onrender.com

Optional env:
    SIGNALFORGE_DISCORD_GUILD_ID=123456789

If SIGNALFORGE_DISCORD_GUILD_ID is set, slash commands sync to that guild
quickly. Without it, Discord global command propagation can take longer.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import discord
import httpx
from discord import app_commands

from app.config import get_settings
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20.0


@contextmanager
def _single_instance_lock():
    lock_file = Path(tempfile.gettempdir()) / "signalforge-discord-bot.lock"
    handle = lock_file.open("w")
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SystemExit("Another SignalForge Discord bot process is already running.") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SystemExit("Another SignalForge Discord bot process is already running.") from exc
        yield
    finally:
        handle.close()


def _api_base() -> str:
    return get_settings().api_url.rstrip("/")


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.0f}"


def _price(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _signal_embed(signal: dict[str, Any]) -> discord.Embed:
    tier = str(signal.get("tier", "high_conviction")).replace("_", " ").upper()
    market = signal.get("market") or "Unknown market"
    side = signal.get("side") or "n/a"
    outcome = signal.get("outcome")
    if outcome:
        side = f"{side} / {outcome}"

    embed = discord.Embed(
        title=f"{tier} - {market}",
        url=signal.get("market_url"),
        color=0xDC2626 if signal.get("tier") == "high_conviction" else 0x16A34A,
    )
    embed.add_field(name="Side", value=side, inline=True)
    embed.add_field(name="Score", value=str(signal.get("score", "n/a")), inline=True)
    if signal.get("event_date"):
        embed.add_field(name="Event date", value=str(signal.get("event_date")), inline=True)
    embed.add_field(name="Chase risk", value=str(signal.get("chase_risk", "n/a")), inline=True)
    embed.add_field(
        name="Smart money",
        value=f"{signal.get('trader_count', 0)} wallets aligned",
        inline=True,
    )
    embed.add_field(
        name="Total tracked size",
        value=_money(signal.get("total_tracked_size")),
        inline=True,
    )
    embed.add_field(
        name="Entry range",
        value=f"{_price(signal.get('entry_price_min'))}-{_price(signal.get('entry_price_max'))}",
        inline=True,
    )
    embed.add_field(name="Current price", value=_price(signal.get("current_price")), inline=True)
    embed.add_field(name="Action", value=str(signal.get("action") or "Watch only"), inline=False)

    reason = str(signal.get("reason") or "No reason provided")
    embed.add_field(name="Reason", value=reason[:1024], inline=False)

    links = []
    if signal.get("market_url"):
        links.append(f"[Market]({signal['market_url']})")
    if signal.get("trader_url") and signal.get("trader"):
        links.append(f"[{signal['trader']}]({signal['trader_url']})")
    if links:
        embed.add_field(name="Links", value=" | ".join(links), inline=False)
    return embed


def _near_miss_embed(signal: dict[str, Any]) -> discord.Embed:
    embed = _signal_embed(signal)
    embed.title = f"NEAR MISS - {signal.get('market') or 'Unknown market'}"
    embed.color = 0xF59E0B
    embed.insert_field_at(
        0,
        name="Why it failed",
        value=str(signal.get("failed_reason") or signal.get("reason") or "n/a")[:1024],
        inline=False,
    )
    return embed


def _position_embed(position: dict[str, Any]) -> discord.Embed:
    market = position.get("market") or "Unknown market"
    trader = position.get("trader") or "Unknown trader"
    side = position.get("side") or "n/a"
    outcome = position.get("outcome")
    if outcome:
        side = f"{side} / {outcome}"

    embed = discord.Embed(
        title=f"{trader} - {side}",
        url=position.get("market_url"),
        color=0x2563EB,
    )
    embed.add_field(name="Market", value=str(market)[:1024], inline=False)
    if position.get("event_date"):
        embed.add_field(name="Event date", value=str(position.get("event_date")), inline=True)
    embed.add_field(name="Score", value=str(position.get("score", "n/a")), inline=True)
    embed.add_field(name="Confidence", value=str(position.get("confidence", "n/a")), inline=True)
    embed.add_field(name="Avg entry", value=_price(position.get("avg_entry_price")), inline=True)
    embed.add_field(name="Total size", value=_money(position.get("total_size_usd")), inline=True)
    embed.add_field(name="Trades", value=str(position.get("trade_count", "n/a")), inline=True)
    if position.get("last_trade_at"):
        embed.add_field(name="Last trade", value=str(position.get("last_trade_at")), inline=False)

    reason = str(position.get("reason") or "No reason available")
    embed.add_field(name="Reason", value=reason[:1024], inline=False)

    links = []
    if position.get("market_url"):
        links.append(f"[Market]({position['market_url']})")
    if position.get("trader_url") and trader:
        links.append(f"[{trader}]({position['trader_url']})")
    if links:
        embed.add_field(name="Links", value=" | ".join(links), inline=False)
    return embed


class SignalForgeBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        settings = get_settings()
        if settings.discord_guild_id:
            guild = discord.Object(id=settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            try:
                await self.tree.sync(guild=guild)
            except discord.Forbidden as exc:
                raise SystemExit(
                    "Discord command sync failed: missing access to guild "
                    f"{settings.discord_guild_id}. Invite the bot to that server "
                    "with the 'bot' and 'applications.commands' scopes, or fix "
                    "SIGNALFORGE_DISCORD_GUILD_ID."
                ) from exc
            logger.info("Synced SignalForge commands to guild %s", settings.discord_guild_id)
        else:
            await self.tree.sync()
            logger.info("Synced SignalForge commands globally")


bot = SignalForgeBot()


@bot.tree.command(name="sf_status", description="Check SignalForge backend, scanner, and alert status.")
async def sf_status(interaction: discord.Interaction) -> None:
    await interaction.response.defer(thinking=True, ephemeral=True)
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(f"{_api_base()}/bot/status")
            response.raise_for_status()
            status = response.json()
    except httpx.HTTPError as exc:
        await interaction.followup.send(f"SignalForge status check failed: `{type(exc).__name__}: {exc}`")
        return

    falcon = status.get("falcon", {})
    embed = discord.Embed(
        title="SignalForge Status",
        description=f"Backend: `{_api_base()}`",
        color=0x16A34A if status.get("status") == "ok" else 0xF59E0B,
    )
    embed.add_field(name="Environment", value=str(status.get("environment")), inline=True)
    embed.add_field(name="Traders", value=str(status.get("traders")), inline=True)
    embed.add_field(name="Markets", value=str(status.get("markets")), inline=True)
    embed.add_field(name="Signals 24h", value=str(status.get("signals_24h")), inline=True)
    embed.add_field(name="Discord sent 24h", value=str(status.get("discord_sent_24h")), inline=True)
    embed.add_field(name="Discord skipped 24h", value=str(status.get("discord_skipped_24h")), inline=True)
    embed.add_field(
        name="Falcon",
        value=(
            f"configured={falcon.get('configured')} healthy={falcon.get('healthy')}\n"
            f"last scan={falcon.get('last_scan_successes')}/{falcon.get('last_scan_calls')}\n"
            f"last error={falcon.get('last_error') or 'none'}"
        )[:1024],
        inline=False,
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(
    name="sf_high_conviction",
    description="Show recent high-conviction or possible-entry SignalForge bets.",
)
@app_commands.describe(
    limit="Number of signals to return, 1-10",
    hours="Lookback window in hours",
    event_date_from="Only include markets dated on/after YYYY-MM-DD",
)
async def sf_high_conviction(
    interaction: discord.Interaction,
    limit: app_commands.Range[int, 1, 10] = 5,
    hours: app_commands.Range[int, 1, 168] = 24,
    event_date_from: str | None = None,
) -> None:
    await interaction.response.defer(thinking=True)
    parsed_event_date: str | None = None
    if event_date_from:
        try:
            parsed_event_date = date.fromisoformat(event_date_from).isoformat()
        except ValueError:
            await interaction.followup.send("`event_date_from` must be in `YYYY-MM-DD` format.")
            return

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            params: dict[str, Any] = {"limit": limit, "hours": hours}
            if parsed_event_date:
                params["event_date_from"] = parsed_event_date
            response = await client.get(
                f"{_api_base()}/bot/high-conviction",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        await interaction.followup.send(f"High-conviction lookup failed: `{type(exc).__name__}: {exc}`")
        return

    signals = payload.get("signals", [])
    if not signals:
        near_misses = payload.get("near_misses", [])
        if not near_misses:
            await interaction.followup.send(
                f"No high-conviction or near-miss signals found in the last {hours}h."
            )
            return
        embeds = [_near_miss_embed(signal) for signal in near_misses[:3]]
        event_text = f" since event date {parsed_event_date}" if parsed_event_date else ""
        await interaction.followup.send(
            content=(
                f"No high-conviction or possible-entry signals found in the last {hours}h"
                f"{event_text}. Top near-misses:"
            ),
            embeds=embeds,
        )
        return

    embeds = [_signal_embed(signal) for signal in signals[:10]]
    await interaction.followup.send(
        content=f"Found {len(embeds)} SignalForge candidate(s) in the last {hours}h.",
        embeds=embeds,
    )


@bot.tree.command(
    name="search",
    description="Search tracked-wallet positions for a Polymarket or Kalshi market URL.",
)
@app_commands.describe(
    market_url="Polymarket or Kalshi market URL",
    limit="Number of positions to return, 1-10",
)
async def search(
    interaction: discord.Interaction,
    market_url: str,
    limit: app_commands.Range[int, 1, 10] = 10,
) -> None:
    await interaction.response.defer(thinking=True)
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                f"{_api_base()}/bot/search",
                params={"market_url": market_url, "limit": limit},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        await interaction.followup.send(f"Market search failed: HTTP {exc.response.status_code} `{detail}`")
        return
    except httpx.HTTPError as exc:
        await interaction.followup.send(f"Market search failed: `{type(exc).__name__}: {exc}`")
        return

    positions = payload.get("positions", [])
    if not positions:
        await interaction.followup.send(
            f"No tracked-wallet positions found for `{payload.get('market_slug') or market_url}`."
        )
        return

    embeds = [_position_embed(position) for position in positions[:10]]
    await interaction.followup.send(
        content=(
            f"Found {len(embeds)} tracked-wallet position(s) for "
            f"{payload.get('market') or payload.get('market_slug')}."
        ),
        embeds=embeds,
    )


def main() -> None:
    configure_logging()
    settings = get_settings()
    if not settings.discord_bot_token:
        raise SystemExit("SIGNALFORGE_DISCORD_BOT_TOKEN is required")
    with _single_instance_lock():
        bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
