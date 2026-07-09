"""
scripts/healthcheck.py
----------------------
Standalone health probe used by Docker ``HEALTHCHECK`` directives and
ops shell scripts.

The script is deliberately decoupled from the agents package: it imports
only stdlib + thin sync clients (httpx, redis, psycopg2) so it can run in
a minimal container or even outside the virtualenv. If the trading code
itself is broken (import errors, settings misconfigured), this probe must
still be able to report what is and isn't reachable.

Exit codes
----------
- 0  — every check passed.
- 1  — at least one check failed (the per-check line tells you which).

Each check prints exactly one line:
    ✓ name
    ✗ name (failure reason)

Output is intentionally simple and grep-friendly so it can be tailed by
log shippers without any parsing logic.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


def _markers() -> tuple[str, str]:
    """
    Return (pass_marker, fail_marker) safe for the current stdout encoding.

    Windows console (cp1252) cannot render the U+2713 / U+2717 glyphs that
    Linux/macOS terminals support. Detect at runtime and fall back to ASCII
    so the probe never crashes on print.
    """
    encoding = (sys.stdout.encoding or "ascii").lower()
    try:
        "✓✗".encode(encoding)
        return "✓", "✗"
    except (UnicodeEncodeError, LookupError):
        return "[OK]", "[FAIL]"


_PASS, _FAIL = _markers()

# ANSI colours, only when stdout is an interactive TTY and not disabled.
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_GREEN = "\033[32m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_DIM = "\033[2m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""

# python-dotenv is in the main deps; if a developer runs this in a fresh
# venv without it we still want the script to work using process env only.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _check_monitoring_http() -> tuple[bool, str]:
    """Reach the MonitoringAgent aiohttp endpoint on port 8081."""
    import httpx

    url = os.environ.get("MONITORING_HEALTH_URL", "http://localhost:8081/health")
    try:
        resp = httpx.get(url, timeout=2.0)
    except Exception as exc:
        return False, f"{url}: {exc}"
    if resp.status_code != 200:
        return False, f"{url}: HTTP {resp.status_code}"
    return True, "monitoring"


def _check_api_http() -> tuple[bool, str]:
    """Reach the FastAPI control plane on port 8000."""
    import httpx

    url = os.environ.get("API_HEALTH_URL", "http://localhost:8000/health")
    try:
        resp = httpx.get(url, timeout=2.0)
    except Exception as exc:
        return False, f"{url}: {exc}"
    if resp.status_code != 200:
        return False, f"{url}: HTTP {resp.status_code}"
    return True, "api"


def _check_redis() -> tuple[bool, str]:
    """PING the Redis instance used by the messaging bus."""
    import redis

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    db = int(os.environ.get("REDIS_DB", "0"))
    password = os.environ.get("REDIS_PASSWORD") or None

    try:
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
        if not client.ping():
            return False, f"{host}:{port}: PING returned falsy"
    except Exception as exc:
        return False, f"{host}:{port}: {exc}"
    return True, "redis"


def _check_postgres() -> tuple[bool, str]:
    """Open a sync connection and run ``SELECT 1``."""
    import psycopg2

    host = os.environ.get("DB_HOST", "localhost")
    port = int(os.environ.get("DB_PORT", "5432"))
    name = os.environ.get("DB_NAME", "trading_db")
    user = os.environ.get("DB_USER", "trading_user")
    password = os.environ.get("DB_PASSWORD", "trading_pass")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
            user=user,
            password=password,
            connect_timeout=2,
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return False, f"{host}:{port}/{name}: {exc}"
    return True, "postgres"


def _check_agents() -> list[tuple[str, bool, str]]:
    """
    Best-effort per-agent rows from the MonitoringAgent registry (port 8081).

    Returns one (name, ok, detail) row per agent that has sent a heartbeat. If
    the monitoring endpoint is unreachable this returns an empty list (the
    monitoring service row already reports the connectivity failure).
    """
    import httpx

    url = os.environ.get("MONITORING_AGENTS_URL", "http://localhost:8081/agents")
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    # The endpoint returns either {"agents": {...}} or a bare mapping.
    agents = data.get("agents", data) if isinstance(data, dict) else {}
    rows: list[tuple[str, bool, str]] = []
    for name, info in sorted(agents.items()):
        if not isinstance(info, dict):
            continue
        status = str(info.get("status", "unknown"))
        stale = bool(info.get("is_stale", False))
        age = info.get("last_seen_seconds_ago")
        ok = status.lower() in ("running", "starting") and not stale
        detail = f"{status}, last seen {age}s ago" if age is not None else status
        rows.append((f"agent:{name}", ok, detail))
    return rows


_CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("redis", _check_redis),
    ("postgres", _check_postgres),
    ("monitoring", _check_monitoring_http),
    ("api", _check_api_http),
]


def _print_table(rows: list[tuple[str, bool, str]]) -> None:
    """Render a green/red status table to stdout."""
    name_w = max((len(n) for n, _, _ in rows), default=8)
    header = f"  {'SERVICE'.ljust(name_w)}   STATUS   DETAIL"
    print(header)
    print(f"  {'-' * name_w}   ------   {'-' * 24}")
    for name, ok, detail in rows:
        colour = _GREEN if ok else _RED
        marker = _PASS if ok else _FAIL
        status = f"{colour}{marker} {'UP' if ok else 'DOWN'}{_RESET}"
        detail_txt = "" if ok and not detail else f"{_DIM}{detail}{_RESET}"
        print(f"  {name.ljust(name_w)}   {status:<6}   {detail_txt}")


def main() -> int:
    """Run every check, print a status table, and return 0 if all pass."""
    rows: list[tuple[str, bool, str]] = []
    overall_ok = True

    for name, check in _CHECKS:
        try:
            ok, detail = check()
        except Exception as exc:
            ok, detail = False, f"unhandled: {exc}"
        if not ok:
            overall_ok = False
        rows.append((name, ok, "" if ok else detail))

    # Per-agent rows (don't flip overall_ok — agents may still be warming up).
    rows.extend(_check_agents())

    _print_table(rows)

    summary_colour = _GREEN if overall_ok else _RED
    print()
    print(
        f"{summary_colour}{'ALL CORE SERVICES UP' if overall_ok else 'DEGRADED - see DOWN rows above'}{_RESET}"
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
