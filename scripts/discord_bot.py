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
from typing import Any

import discord
import httpx
from discord import app_commands

from app.config import get_settings
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)

API_BASE = os.environ.get("SIGNALFORGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 20.0


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
            await self.tree.sync(guild=guild)
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
            response = await client.get(f"{API_BASE}/bot/status")
            response.raise_for_status()
            status = response.json()
    except httpx.HTTPError as exc:
        await interaction.followup.send(f"SignalForge status check failed: `{type(exc).__name__}: {exc}`")
        return

    falcon = status.get("falcon", {})
    embed = discord.Embed(
        title="SignalForge Status",
        description=f"Backend: `{API_BASE}`",
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
@app_commands.describe(limit="Number of signals to return, 1-10", hours="Lookback window in hours")
async def sf_high_conviction(
    interaction: discord.Interaction,
    limit: app_commands.Range[int, 1, 10] = 5,
    hours: app_commands.Range[int, 1, 168] = 24,
) -> None:
    await interaction.response.defer(thinking=True)
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE}/bot/high-conviction",
                params={"limit": limit, "hours": hours},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        await interaction.followup.send(f"High-conviction lookup failed: `{type(exc).__name__}: {exc}`")
        return

    signals = payload.get("signals", [])
    if not signals:
        await interaction.followup.send(
            f"No high-conviction or possible-entry signals found in the last {hours}h."
        )
        return

    embeds = [_signal_embed(signal) for signal in signals[:10]]
    await interaction.followup.send(
        content=f"Found {len(embeds)} SignalForge candidate(s) in the last {hours}h.",
        embeds=embeds,
    )


def main() -> None:
    configure_logging()
    settings = get_settings()
    if not settings.discord_bot_token:
        raise SystemExit("SIGNALFORGE_DISCORD_BOT_TOKEN is required")
    bot.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
