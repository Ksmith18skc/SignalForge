"""BallparkPal provider + cache tests.

We never hit the live site here. Every test feeds a small fixture HTML
snippet through the pure parsers, or exercises the read/write helpers
against the in-memory SQLite session provided by ``conftest``.
"""

from __future__ import annotations

import os

import pytest

bs4 = pytest.importorskip("bs4", reason="beautifulsoup4 required for BallparkPal tests")

from app.providers.ballparkpal import (  # noqa: E402
    extract_last_updated,
    extract_table_rows,
    looks_like_login_page,
    normalize_team_abbr,
    parse_game_simulations,
    parse_hits,
    parse_home_run_zone,
    parse_page,
    parse_positive_ev,
    parse_strikeout_center,
)
from app.services.ballparkpal_cache import (  # noqa: E402
    latest_snapshot,
    snapshot_payload,
    upsert_snapshot,
)
from app.services.ballparkpal_integration import compare_totals  # noqa: E402


POSITIVE_EV_HTML = """
<html><body>
  <div>Last Updated: 12:10 AM</div>
  <table>
    <tr>
      <th>Tm</th><th>Player</th><th>Bk</th><th>Market</th><th>O/U</th>
      <th>Line</th><th>Odds</th><th>CS</th><th>Δ</th><th>BP</th><th>Δ</th>
    </tr>
    <tr>
      <td>BOS</td><td>Payton Tolle</td><td>NVG</td><td>Outs</td><td>O</td>
      <td>16.5</td><td>133</td><td>-110</td><td>9.6%</td><td>-130</td><td>13.7%</td>
    </tr>
    <tr>
      <td>LAA</td><td>Vaughn Grissom</td><td>TSC</td><td>Hits</td><td>O</td>
      <td>0.5</td><td>-200</td><td>-268</td><td>6.2%</td><td>-156</td><td>-5.8%</td>
    </tr>
  </table>
</body></html>
"""


STRIKEOUT_HTML = """
<html><body>
  <div>Last Updated: 12:06 AM</div>
  <table>
    <tr>
      <th>Team</th><th>Pitcher</th><th>K</th><th>Opp</th>
      <th>Inn</th><th>BF</th><th>Over</th><th>BP</th><th>KA</th>
    </tr>
    <tr>
      <td>ATL</td><td>Chris Sale</td><td>6.16</td><td>BOS</td>
      <td>5.8</td><td>23.6</td><td>6.5</td><td>+130</td><td></td>
    </tr>
    <tr>
      <td>PIT</td><td>Paul Skenes</td><td>6.15</td><td>CHC</td>
      <td>6.1</td><td>24.3</td><td>6.5</td><td>+133</td><td></td>
    </tr>
  </table>
</body></html>
"""


HR_ZONE_HTML = """
<html><body>
  <div>Last Updated: 12:06 AM</div>
  <table>
    <tr><th>Park</th><th>Total</th></tr>
    <tr><td>Globe Life Field</td><td>2.29</td></tr>
    <tr><td>Rate Field</td><td>2.17</td></tr>
  </table>
  <table>
    <tr><th>Away</th><th>HRs</th><th>Home</th><th>HRs</th></tr>
    <tr><td>HOU</td><td>1.01</td><td>TEX</td><td>1.28</td></tr>
  </table>
</body></html>
"""


HITS_HTML = """
<html><body>
  <div>Last Updated: 12:10 AM</div>
  <table>
    <tr>
      <th>Player</th><th>Team</th><th>Opp</th>
      <th>Projected Hits</th><th>Line</th>
      <th>Over</th><th>Under</th>
    </tr>
    <tr>
      <td>Eli White</td><td>ATL</td><td>BOS</td>
      <td>1.10</td><td>0.5</td><td>-109</td><td>+102</td>
    </tr>
  </table>
</body></html>
"""


GAME_SIMS_HTML = """
<html><body>
  <div>Last Updated: 12:10 AM</div>
  <table>
    <tr>
      <th>Away</th><th>Home</th>
      <th>Away Runs</th><th>Home Runs</th>
      <th>Total</th><th>Away WP</th><th>Home WP</th>
    </tr>
    <tr>
      <td>HOU</td><td>TEX</td>
      <td>4.2</td><td>4.6</td><td>8.8</td><td>45.3%</td><td>54.7%</td>
    </tr>
  </table>
</body></html>
"""


LOGIN_HTML = """
<html><body>
  <form>
    <input type="text" name="email"/>
    <input type="password" name="pw"/>
    <button>Sign In</button>
    <a href="/Forgot-Password.php">Forgot password</a>
  </form>
</body></html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_team_abbr_aliases():
    assert normalize_team_abbr("CHW") == "CHW"
    assert normalize_team_abbr("WSX") == "CHW"
    assert normalize_team_abbr("TBR") == "TB"
    assert normalize_team_abbr("") is None
    assert normalize_team_abbr(None) is None
    # Unknown abbreviations pass through untouched so the parser doesn't
    # silently corrupt a row by guessing.
    assert normalize_team_abbr("ZZZ") == "ZZZ"


def test_extract_last_updated_returns_banner_text():
    assert extract_last_updated("<div>Last Updated: 12:10 AM</div>") == "12:10 AM"
    assert extract_last_updated("") is None
    assert extract_last_updated("<div>no banner here</div>") is None


def test_extract_table_rows_skips_tables_without_headers():
    html = "<table><tr><td>a</td><td>b</td></tr></table>"
    # No <th> → no usable headers → table is skipped.
    assert extract_table_rows(html) == []


def test_looks_like_login_page_detects_form_and_url():
    assert looks_like_login_page(url="https://www.ballparkpal.com/Login.php", html="<html/>")
    assert looks_like_login_page(url="https://www.ballparkpal.com/Positive-EV.php", html=LOGIN_HTML)
    assert not looks_like_login_page(
        url="https://www.ballparkpal.com/Positive-EV.php", html=POSITIVE_EV_HTML,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_positive_ev_extracts_rows():
    result = parse_positive_ev(POSITIVE_EV_HTML)
    rows = result["rows"]
    assert len(rows) == 2
    first = rows[0]
    assert first["team"] == "BOS"
    assert first["player"] == "Payton Tolle"
    assert first["book"] == "NVG"
    assert first["market"] == "Outs"
    assert first["line"] == 16.5
    assert first["odds"] == 133
    assert first["consensus_odds"] == -110
    assert first["ballparkpal_delta"] == 13.7  # "13.7%" → 13.7
    assert result["meta"]["last_updated"] == "12:10 AM"


def test_parse_positive_ev_handles_empty_html():
    out = parse_positive_ev("<html><body></body></html>")
    assert out["rows"] == []
    assert any("No tables" in w for w in out["warnings"])


def test_parse_strikeout_center_extracts_rows():
    out = parse_strikeout_center(STRIKEOUT_HTML)
    rows = out["rows"]
    assert len(rows) == 2
    assert rows[0]["pitcher"] == "Chris Sale"
    assert rows[0]["projected_k"] == 6.16
    assert rows[0]["projected_innings"] == 5.8
    assert rows[0]["batters_faced"] == 23.6


def test_parse_home_run_zone_separates_totals_and_by_game():
    out = parse_home_run_zone(HR_ZONE_HTML)
    meta = out["meta"]
    assert any(r["park"] == "Globe Life Field" for r in meta["totals"])
    assert meta["totals"][0]["total_projected_hrs"] == 2.29
    assert meta["by_game"][0]["home_team"] == "TEX"
    assert meta["by_game"][0]["away_projected_hrs"] == 1.01


def test_parse_hits_extracts_row():
    out = parse_hits(HITS_HTML)
    row = out["rows"][0]
    assert row["player"] == "Eli White"
    assert row["projected_hits"] == 1.1
    assert row["line"] == 0.5
    assert row["over_odds"] == -109


def test_parse_game_simulations_extracts_total():
    out = parse_game_simulations(GAME_SIMS_HTML)
    row = out["rows"][0]
    assert row["away_team"] == "HOU"
    assert row["home_team"] == "TEX"
    assert row["projected_total"] == 8.8
    assert row["win_probability_home"] == 54.7


def test_parse_page_routes_to_correct_parser():
    assert parse_page("positive_ev", POSITIVE_EV_HTML)["rows"]
    # Unknown page returns empty result with a warning instead of crashing.
    out = parse_page("nope", POSITIVE_EV_HTML)
    assert out["rows"] == []
    assert any("Unknown" in w for w in out["warnings"])


def test_parse_page_swallows_parser_exceptions(monkeypatch):
    # Force the chosen parser to raise; parse_page must convert it into a
    # structured warning so a single bad page can't kill the whole run.
    import app.providers.ballparkpal as bpp

    def _boom(html):
        raise RuntimeError("synthetic")

    monkeypatch.setitem(bpp.PARSERS, "positive_ev", _boom)
    out = bpp.parse_page("positive_ev", POSITIVE_EV_HTML)
    assert out["rows"] == []
    assert any("Parser exception" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Cache layer (DB)
# ---------------------------------------------------------------------------


def test_upsert_snapshot_inserts_then_updates(db_session):
    parsed = parse_positive_ev(POSITIVE_EV_HTML)
    first = upsert_snapshot(
        db_session,
        page="positive_ev",
        slate_date="2026-05-28",
        source_url="https://www.ballparkpal.com/Positive-EV.php",
        parsed=parsed,
        raw_html_path=".cache/test.html",
        last_updated_text="12:10 AM",
        status="ok",
    )
    db_session.commit()
    assert first.id is not None
    assert first.row_count == 2

    parsed2 = parse_positive_ev(POSITIVE_EV_HTML * 1)  # same content
    second = upsert_snapshot(
        db_session,
        page="positive_ev",
        slate_date="2026-05-28",
        source_url="https://www.ballparkpal.com/Positive-EV.php",
        parsed=parsed2,
        raw_html_path=".cache/test.html",
        last_updated_text="12:11 AM",
        status="ok",
    )
    db_session.commit()
    # Same (page, slate_date) → same row, no duplicate inserted.
    assert second.id == first.id
    assert second.last_updated_text == "12:11 AM"


def test_snapshot_payload_renders_missing_state():
    payload = snapshot_payload(None)
    assert payload["status"] == "missing"
    assert payload["rows"] == []
    assert payload["stale"] is True


def test_latest_snapshot_returns_newest(db_session):
    parsed = parse_positive_ev(POSITIVE_EV_HTML)
    upsert_snapshot(
        db_session,
        page="positive_ev",
        slate_date="2026-05-27",
        source_url="x",
        parsed=parsed,
        raw_html_path=None,
        last_updated_text=None,
        status="ok",
    )
    upsert_snapshot(
        db_session,
        page="positive_ev",
        slate_date="2026-05-28",
        source_url="x",
        parsed=parsed,
        raw_html_path=None,
        last_updated_text=None,
        status="ok",
    )
    db_session.commit()
    snap = latest_snapshot(db_session, page="positive_ev")
    assert snap is not None
    assert snap.slate_date == "2026-05-28"


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


def test_job_payload_handles_missing_job():
    from app.services.ballparkpal_jobs import job_payload

    payload = job_payload(None)
    assert payload["status"] == "missing"
    assert payload["job_id"] is None


def test_start_refresh_rejects_concurrent_run(db_session, monkeypatch):
    """Two simultaneous refresh requests must not both spawn a subprocess —
    the second one raises BallparkPalBusyError and is surfaced to the API
    as a 409.
    """
    from app.services import ballparkpal_jobs as jobs

    # Stub subprocess.Popen with a fake that "stays running" — the watcher
    # would normally finalize the job; we want it left active so the
    # concurrency guard fires.
    class _FakePopen:
        def __init__(self, *a, **kw):
            self.pid = 99999
            self.stdin = type("S", (), {"write": lambda *a, **k: None,
                                          "flush": lambda *a, **k: None,
                                          "close": lambda *a, **k: None,
                                          "closed": False})()
            self.stdout = type(
                "O",
                (),
                {"readline": lambda *a, **k: b""},
            )()
            self._poll_calls = 0

        def poll(self):
            # Stay alive forever in this test so the watcher loops and the
            # job stays "running" long enough for the guard to fire.
            return None

        def wait(self, timeout=None):  # noqa: ARG002
            return None

        @property
        def returncode(self):
            return None

    # Patch the spawn helper so we don't actually launch a Python child.
    monkeypatch.setattr(
        jobs, "_spawn_process", lambda job_id, spec: (_FakePopen(), None),
    )
    # Make the watcher a no-op so the row stays "running".
    monkeypatch.setattr(jobs, "_watch", lambda *a, **k: None)
    # And keep the singleton ACTIVE_HANDLES table writable by the test.
    jobs._ACTIVE_HANDLES.clear()

    first = jobs.start_refresh(db_session, pages=["positive_ev"])
    assert first.status in {"queued", "running"}
    with pytest.raises(jobs.BallparkPalBusyError):
        jobs.start_refresh(db_session, pages=["positive_ev"])

    # Cleanup: clear handles so we don't leak state into later tests.
    jobs._ACTIVE_HANDLES.clear()


def test_spawn_process_sets_cwd_and_pythonpath_to_repo_root(monkeypatch):
    """Regression: the dashboard-triggered subprocess must launch with
    cwd=repo_root and PYTHONPATH prefixed by repo_root so the child can
    ``import app`` regardless of where Streamlit was started.
    """
    import subprocess as _sp

    from app.services import ballparkpal_jobs as jobs

    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(self, cmd, *, cwd, stdin, stdout, stderr, env, bufsize):  # noqa: ARG002
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["env"] = env
            self.pid = 4242
            self.stdin = None
            self.stdout = None

        def poll(self):
            return 0

        @property
        def returncode(self):
            return 0

    monkeypatch.setattr(_sp, "Popen", _FakePopen)
    monkeypatch.setattr(jobs.subprocess, "Popen", _FakePopen)
    # CLI_SCRIPT must exist on disk (it does in this repo) for the
    # early-exit path not to fire. Assert it so the regression test is
    # explicit about the precondition.
    assert jobs.CLI_SCRIPT.exists(), f"CLI script missing: {jobs.CLI_SCRIPT}"

    spec = jobs.JobSpec(
        mode="refresh", pages=["positive_ev"], slate_date=None,
        headless=True, timeout_seconds=60,
    )
    popen, error = jobs._spawn_process("test-job", spec)
    assert popen is not None and error is None
    assert captured["cwd"] == str(jobs.REPO_ROOT)
    env = captured["env"]
    assert isinstance(env, dict)
    pp = env.get("PYTHONPATH", "")
    # The very first PYTHONPATH segment must be repo_root so the child's
    # `import app` resolves before any inherited PYTHONPATH entries.
    first_segment = pp.split(os.pathsep, 1)[0] if pp else ""
    assert first_segment == str(jobs.REPO_ROOT), (
        f"PYTHONPATH first segment was {first_segment!r}, expected {str(jobs.REPO_ROOT)!r}"
    )


def test_start_refresh_marks_failed_when_cli_missing(db_session, monkeypatch):
    """If the spawn helper returns no popen, the job must be finalized as
    failed in the same request — never left dangling as 'queued'.
    """
    from app.services import ballparkpal_jobs as jobs

    monkeypatch.setattr(
        jobs, "_spawn_process", lambda job_id, spec: (None, "CLI script missing"),
    )
    jobs._ACTIVE_HANDLES.clear()
    job = jobs.start_refresh(db_session, pages=["positive_ev"])
    assert job.status == "failed"
    assert "CLI script missing" in (job.error_message or "")


def test_compare_totals_marks_agreement_and_disagreement():
    agree = compare_totals(
        signalforge_projected_total=9.0,
        market_total=9.5,
        ballparkpal_total=9.2,
    )
    assert agree["verdict"] == "agree"
    inflated = compare_totals(
        signalforge_projected_total=10.5,
        market_total=8.5,
        ballparkpal_total=9.0,
    )
    assert inflated["verdict"] == "signalforge_inflated_vs_bpp"
    none = compare_totals(
        signalforge_projected_total=None,
        market_total=8.5,
        ballparkpal_total=None,
    )
    assert none["verdict"] == "no_comparison"
