from __future__ import annotations

import pytest

pytest.importorskip("discord")

from scripts import discord_bot


class _FakeResponse:
    def __init__(self) -> None:
        self.deferred = False
        self.sent = []

    def is_done(self) -> bool:
        return self.deferred or bool(self.sent)

    async def defer(self, *, thinking: bool = False, ephemeral: bool = False) -> None:
        self.deferred = True
        self.thinking = thinking
        self.ephemeral = ephemeral

    async def send_message(self, *args, **kwargs) -> None:
        self.sent.append((args, kwargs))


class _FakeFollowup:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, *args, **kwargs) -> None:
        self.sent.append((args, kwargs))


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()
        self.command = "test"


@pytest.mark.asyncio
async def test_status_command_defers_before_backend_call(monkeypatch):
    interaction = _FakeInteraction()

    async def fake_get_json(path, params=None):
        assert path == "/bot/status"
        assert params is None
        assert interaction.response.deferred is True
        return {
            "status": "ok",
            "environment": "test",
            "traders": 0,
            "markets": 0,
            "signals_24h": 0,
            "discord_sent_24h": 0,
            "discord_skipped_24h": 0,
            "falcon": {},
        }

    monkeypatch.setattr(discord_bot, "_get_json", fake_get_json)
    monkeypatch.setattr(discord_bot, "_api_base", lambda: "http://api.test")

    await discord_bot.sf_status.callback(interaction)

    assert interaction.response.deferred is True
    assert interaction.response.ephemeral is True
    assert interaction.followup.sent


@pytest.mark.asyncio
async def test_global_command_error_sends_response_without_prior_defer():
    interaction = _FakeInteraction()

    await discord_bot.on_app_command_error(interaction, RuntimeError("boom"))

    assert interaction.response.sent
    args, kwargs = interaction.response.sent[0]
    assert "SignalForge command failed" in args[0]
    assert kwargs["ephemeral"] is True
