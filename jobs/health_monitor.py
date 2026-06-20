"""
Phase 7 — System Health Monitor.

Runs every N minutes. Checks all pipeline components, persists status to DB,
and fires Telegram alerts on state transitions (UP→DOWN, DOWN→UP).

Required components (trading halts if DOWN):
  database, gamma_api, llm_anthropic, scanner_job

Optional (alert-only):
  coingecko, fear_greed, resolution_checker, risk_engine, opportunities

Live-mode required (if PAPER_TRADING=False):
  polygon_rpc, clob_api, wallet_balance
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import Settings
from notifications.telegram import TelegramAlerter

logger = logging.getLogger(__name__)

# Categories for grouping in the UI
_CAT_REQUIRED = "required"
_CAT_DATA = "data"
_CAT_JOBS = "jobs"
_CAT_MODEL = "model"
_CAT_LIVE = "live"

_UP = "UP"
_DOWN = "DOWN"
_DEGRADED = "DEGRADED"
_UNKNOWN = "UNKNOWN"
_NA = "N/A"


def _status_entry(
    name: str,
    display_name: str,
    category: str,
    status: str,
    detail: str,
    required: bool = False,
    latency_ms: float | None = None,
) -> dict:
    return {
        "name": name,
        "display_name": display_name,
        "category": category,
        "status": status,
        "detail": detail,
        "required": required,
        "latency_ms": round(latency_ms, 0) if latency_ms is not None else None,
        "checked_at": datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC"),
    }


class HealthMonitor:
    def __init__(
        self,
        session_factory: async_sessionmaker,
        settings: Settings,
        alerter: TelegramAlerter,
        executor=None,   # TradeExecutor reference for wallet/CLOB checks
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._alerter = alerter
        self._executor = executor

        # Previous statuses for transition detection
        self._prev: dict[str, str] = {}
        # In-memory cache served to dashboard
        self._statuses: list[dict] = []
        # Circuit breaker state for alert dedup
        self._circuit_was_tripped: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def get_statuses(self) -> list[dict]:
        return list(self._statuses)

    def is_trading_halted(self) -> bool:
        """True if any required component is DOWN. Empty statuses → allow (startup)."""
        if not self._statuses:
            return False
        return any(
            s["required"] and s["status"] == _DOWN
            for s in self._statuses
        )

    def get_halt_reasons(self) -> list[str]:
        return [
            s["display_name"]
            for s in self._statuses
            if s["required"] and s["status"] == _DOWN
        ]

    async def run(self) -> None:
        checks = await asyncio.gather(
            self._check_database(),
            self._check_gamma_api(),
            self._check_llm(),
            self._check_scanner_job(),
            self._check_coingecko(),
            self._check_fear_greed(),
            self._check_resolution_checker(),
            self._check_risk_engine(),
            self._check_opportunities(),
            return_exceptions=True,
        )

        # Add live-mode checks if applicable
        if self._executor and not self._settings.paper_trading:
            live_checks = await asyncio.gather(
                self._check_polygon_rpc(),
                self._check_clob_api(),
                self._check_wallet_balance(),
                return_exceptions=True,
            )
            checks = checks + live_checks
        else:
            # Always show these rows, marked N/A in paper mode
            checks = list(checks) + [
                _status_entry("polygon_rpc", "Polygon RPC", _CAT_LIVE, _NA,
                              "Paper mode — not connected", False),
                _status_entry("clob_api", "CLOB API", _CAT_LIVE, _NA,
                              "Paper mode — not connected", False),
                _status_entry("wallet_balance", "Wallet Balance", _CAT_LIVE, _NA,
                              "Set POLYGON_PRIVATE_KEY to enable", False),
            ]

        statuses = [s for s in checks if isinstance(s, dict)]

        # Fire Telegram alerts on state transitions
        await self._process_transitions(statuses)

        # Check circuit breaker separately and alert on flip
        await self._check_circuit_breaker_state()

        # Persist to DB
        await self._persist(statuses)
        self._statuses = statuses

    # ── Individual checks ─────────────────────────────────────────────────────

    async def _check_database(self) -> dict:
        t0 = time.perf_counter()
        try:
            from db import queries
            async with self._session_factory() as session:
                await queries.get_last_scan_time(session)
            ms = (time.perf_counter() - t0) * 1000
            return _status_entry("database", "Database", _CAT_REQUIRED, _UP,
                                 f"Connected — {ms:.0f}ms", required=True, latency_ms=ms)
        except Exception as exc:
            return _status_entry("database", "Database", _CAT_REQUIRED, _DOWN,
                                 f"Query failed: {exc}", required=True)

    async def _check_gamma_api(self) -> dict:
        t0 = time.perf_counter()
        try:
            # Also check DB freshness
            from db import queries
            async with self._session_factory() as session:
                last = await queries.get_last_scan_time(session)

            stale_threshold = timedelta(minutes=self._settings.scan_interval_minutes * 2.5)
            if last is None:
                detail = "No scans recorded yet"
                status = _UNKNOWN
            elif datetime.now(tz=timezone.utc) - last > stale_threshold:
                age = datetime.now(tz=timezone.utc) - last
                detail = f"Data stale — last snapshot {int(age.total_seconds()//60)}m ago"
                status = _DEGRADED
            else:
                age = datetime.now(tz=timezone.utc) - last
                ms = (time.perf_counter() - t0) * 1000
                detail = f"Last scan {int(age.total_seconds()//60)}m ago — {ms:.0f}ms"
                status = _UP

            # Quick connectivity ping
            if status != _DOWN:
                async with httpx.AsyncClient(timeout=8) as client:
                    resp = await client.get(
                        f"{self._settings.gamma_base_url}/markets",
                        params={"limit": 1},
                    )
                    if resp.status_code != 200:
                        status = _DOWN
                        detail = f"API returned HTTP {resp.status_code}"

            return _status_entry("gamma_api", "Gamma API", _CAT_REQUIRED, status,
                                 detail, required=(status == _DOWN))
        except Exception as exc:
            return _status_entry("gamma_api", "Gamma API", _CAT_REQUIRED, _DOWN,
                                 f"Unreachable: {exc}", required=True)

    async def _check_llm(self) -> dict:
        try:
            from db import queries
            from db.models import ProbabilityEstimate as PE
            from sqlalchemy import select, func

            async with self._session_factory() as session:
                result = await session.execute(
                    select(func.max(PE.estimated_at))
                )
                last_est = result.scalar_one_or_none()

            if not self._settings.anthropic_api_key:
                return _status_entry("llm_anthropic", "Anthropic LLM", _CAT_REQUIRED,
                                     _DOWN, "ANTHROPIC_API_KEY not configured", required=True)

            if last_est is None:
                status, detail = _UNKNOWN, "No estimates yet — will confirm on first scan"
            else:
                age = datetime.now(tz=timezone.utc) - last_est
                stale = timedelta(minutes=self._settings.scan_interval_minutes * 5)
                if age > stale:
                    status = _DEGRADED
                    detail = f"Last estimate {int(age.total_seconds()//3600)}h ago"
                else:
                    status = _UP
                    detail = f"Last estimate {int(age.total_seconds()//60)}m ago"

            return _status_entry("llm_anthropic", "Anthropic LLM", _CAT_REQUIRED,
                                 status, detail, required=(status == _DOWN))
        except Exception as exc:
            return _status_entry("llm_anthropic", "Anthropic LLM", _CAT_REQUIRED,
                                 _DOWN, str(exc), required=True)

    async def _check_scanner_job(self) -> dict:
        try:
            from db import queries
            async with self._session_factory() as session:
                last = await queries.get_last_scan_time(session)

            if last is None:
                return _status_entry("scanner_job", "Scanner Job", _CAT_JOBS,
                                     _UNKNOWN, "No scans yet", required=False)

            age = datetime.now(tz=timezone.utc) - last
            limit = timedelta(minutes=self._settings.scan_interval_minutes * 3)
            if age > limit:
                return _status_entry("scanner_job", "Scanner Job", _CAT_JOBS,
                                     _DOWN, f"Stale — last run {int(age.total_seconds()//60)}m ago",
                                     required=True)
            return _status_entry("scanner_job", "Scanner Job", _CAT_JOBS,
                                 _UP, f"Last run {int(age.total_seconds()//60)}m ago  "
                                      f"(every {self._settings.scan_interval_minutes}m)")
        except Exception as exc:
            return _status_entry("scanner_job", "Scanner Job", _CAT_JOBS,
                                 _DOWN, str(exc), required=True)

    async def _check_coingecko(self) -> dict:
        t0 = time.perf_counter()
        try:
            from db.models import PriceSnapshot
            from sqlalchemy import select, func
            async with self._session_factory() as session:
                result = await session.execute(select(func.max(PriceSnapshot.scanned_at)))
                last = result.scalar_one_or_none()

            ms = (time.perf_counter() - t0) * 1000
            if last is None:
                return _status_entry("coingecko", "CoinGecko API", _CAT_DATA,
                                     _UNKNOWN, "No price data yet")
            age = datetime.now(tz=timezone.utc) - last
            if age > timedelta(hours=2):
                return _status_entry("coingecko", "CoinGecko API", _CAT_DATA,
                                     _DEGRADED, f"Stale — {int(age.total_seconds()//60)}m ago")
            return _status_entry("coingecko", "CoinGecko API", _CAT_DATA,
                                 _UP, f"Last price {int(age.total_seconds()//60)}m ago",
                                 latency_ms=ms)
        except Exception as exc:
            return _status_entry("coingecko", "CoinGecko API", _CAT_DATA, _DOWN, str(exc))

    async def _check_fear_greed(self) -> dict:
        try:
            from db.models import SentimentSnapshot
            from sqlalchemy import select, func
            async with self._session_factory() as session:
                result = await session.execute(select(func.max(SentimentSnapshot.scanned_at)))
                last = result.scalar_one_or_none()

            if last is None:
                return _status_entry("fear_greed", "Fear & Greed", _CAT_DATA,
                                     _UNKNOWN, "No sentiment data yet")
            age = datetime.now(tz=timezone.utc) - last
            if age > timedelta(hours=6):
                return _status_entry("fear_greed", "Fear & Greed", _CAT_DATA,
                                     _DEGRADED, f"Stale — {int(age.total_seconds()//3600):.0f}h ago")
            return _status_entry("fear_greed", "Fear & Greed", _CAT_DATA,
                                 _UP, f"Last reading {int(age.total_seconds()//60)}m ago")
        except Exception as exc:
            return _status_entry("fear_greed", "Fear & Greed", _CAT_DATA, _DOWN, str(exc))

    async def _check_resolution_checker(self) -> dict:
        try:
            from db.models import PaperPosition
            from sqlalchemy import select, func
            async with self._session_factory() as session:
                result = await session.execute(
                    select(func.max(PaperPosition.closed_at))
                )
                last_close = result.scalar_one_or_none()

            if last_close is None:
                return _status_entry("resolution_checker", "Resolution Checker", _CAT_JOBS,
                                     _UP, f"Running every {self._settings.resolution_check_interval_minutes}m — no resolutions yet")
            age = datetime.now(tz=timezone.utc) - last_close
            return _status_entry("resolution_checker", "Resolution Checker", _CAT_JOBS,
                                 _UP, f"Last resolution {int(age.total_seconds()//3600):.0f}h ago  "
                                      f"(runs every {self._settings.resolution_check_interval_minutes}m)")
        except Exception as exc:
            return _status_entry("resolution_checker", "Resolution Checker", _CAT_JOBS,
                                 _DEGRADED, str(exc))

    async def _check_risk_engine(self) -> dict:
        try:
            from db import queries
            from risk.engine import RiskEngine
            engine = RiskEngine(self._settings)
            async with self._session_factory() as session:
                status_data = await engine.get_status(session)

            if status_data["circuit_tripped"]:
                return _status_entry("risk_engine", "Risk Engine", _CAT_MODEL,
                                     _DEGRADED,
                                     f"Circuit breaker TRIPPED — P&L ${status_data['total_pnl']:+.2f}")
            open_pos = status_data["open_positions"]
            max_pos = status_data["max_open_positions"]
            exp = status_data["total_exposure_usd"]
            return _status_entry("risk_engine", "Risk Engine", _CAT_MODEL,
                                 _UP,
                                 f"OK — {open_pos}/{max_pos} positions  ${exp:.0f} deployed")
        except Exception as exc:
            return _status_entry("risk_engine", "Risk Engine", _CAT_MODEL, _DOWN, str(exc))

    async def _check_opportunities(self) -> dict:
        try:
            from db import queries
            async with self._session_factory() as session:
                opps = await queries.get_active_opportunities(session, hours=24)
                total_estimates = await queries.get_recent_estimate_count(session, hours=24)

            if total_estimates == 0:
                return _status_entry("opportunities", "Opportunity Engine", _CAT_MODEL,
                                     _UNKNOWN, "No LLM estimates in last 24h")
            opp_count = len(opps)
            if opp_count > 0:
                best = max(opps, key=lambda e: abs(e.net_edge))
                direction = "YES" if best.signal == "BUY_YES" else "NO"
                detail = (f"{opp_count} opportunities (last 24h) — best: "
                          f"BUY_{direction} edge {best.net_edge:+.1%}")
            else:
                detail = f"0 opportunities from {total_estimates} estimates (last 24h) — edge threshold may be high"
            return _status_entry("opportunities", "Opportunity Engine", _CAT_MODEL, _UP, detail)
        except Exception as exc:
            return _status_entry("opportunities", "Opportunity Engine", _CAT_MODEL, _DEGRADED, str(exc))

    async def _check_polygon_rpc(self) -> dict:
        t0 = time.perf_counter()
        try:
            wallet = self._executor._wallet if self._executor else None
            if wallet is None:
                return _status_entry("polygon_rpc", "Polygon RPC", _CAT_LIVE,
                                     _DOWN, "Wallet not initialised", required=True)
            connected = await asyncio.to_thread(wallet.is_connected)
            ms = (time.perf_counter() - t0) * 1000
            if not connected:
                return _status_entry("polygon_rpc", "Polygon RPC", _CAT_LIVE,
                                     _DOWN, "RPC unreachable", required=True)
            return _status_entry("polygon_rpc", "Polygon RPC", _CAT_LIVE,
                                 _UP, f"Connected — {ms:.0f}ms", required=True, latency_ms=ms)
        except Exception as exc:
            return _status_entry("polygon_rpc", "Polygon RPC", _CAT_LIVE,
                                 _DOWN, str(exc), required=True)

    async def _check_clob_api(self) -> dict:
        t0 = time.perf_counter()
        try:
            clob = self._executor._clob if self._executor else None
            if clob is None:
                return _status_entry("clob_api", "CLOB API", _CAT_LIVE,
                                     _DOWN, "CLOB not initialised", required=True)
            ok = await asyncio.to_thread(clob.health_check)
            ms = (time.perf_counter() - t0) * 1000
            if not ok:
                return _status_entry("clob_api", "CLOB API", _CAT_LIVE,
                                     _DOWN, "Health check failed", required=True)
            return _status_entry("clob_api", "CLOB API", _CAT_LIVE,
                                 _UP, f"Connected — {ms:.0f}ms", required=True, latency_ms=ms)
        except Exception as exc:
            return _status_entry("clob_api", "CLOB API", _CAT_LIVE,
                                 _DOWN, str(exc), required=True)

    async def _check_wallet_balance(self) -> dict:
        try:
            wallet = self._executor._wallet if self._executor else None
            if wallet is None:
                return _status_entry("wallet_balance", "Wallet Balance", _CAT_LIVE,
                                     _DOWN, "Wallet not initialised", required=True)
            usdc, pol = await asyncio.gather(
                asyncio.to_thread(wallet.usdc_balance),
                asyncio.to_thread(wallet.pol_balance),
            )
            # Store balance snapshot
            await self._store_balance_snapshot(usdc, pol)

            threshold = self._settings.min_usdc_alert_usd
            if threshold > 0 and usdc < threshold:
                await self._alerter.alert_low_balance(usdc, threshold)
                detail = f"LOW — ${usdc:.2f} USDC / {pol:.4f} POL (below ${threshold:.0f} threshold)"
                return _status_entry("wallet_balance", "Wallet Balance", _CAT_LIVE,
                                     _DEGRADED, detail)

            # 24h balance change
            delta_str = await self._balance_delta_24h(usdc)
            detail = f"${usdc:.2f} USDC / {pol:.4f} POL{delta_str}"
            return _status_entry("wallet_balance", "Wallet Balance", _CAT_LIVE, _UP, detail)
        except Exception as exc:
            return _status_entry("wallet_balance", "Wallet Balance", _CAT_LIVE,
                                 _DOWN, str(exc), required=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _process_transitions(self, statuses: list[dict]) -> None:
        for s in statuses:
            name = s["name"]
            curr = s["status"]
            prev = self._prev.get(name, _UNKNOWN)

            if curr in (_DOWN, _DEGRADED) and prev not in (_DOWN, _DEGRADED):
                await self._alerter.alert_component_down(s["display_name"], s["detail"])
            elif curr == _UP and prev in (_DOWN, _DEGRADED):
                await self._alerter.alert_component_recovered(s["display_name"])

            self._prev[name] = curr

    async def _check_circuit_breaker_state(self) -> None:
        try:
            from risk.engine import RiskEngine
            from db import queries
            engine = RiskEngine(self._settings)
            async with self._session_factory() as session:
                status_data = await engine.get_status(session)
            tripped = status_data["circuit_tripped"]
            total_pnl = status_data["total_pnl"]
            threshold = status_data["circuit_threshold_usd"]

            if tripped and not self._circuit_was_tripped:
                await self._alerter.alert_circuit_breaker_tripped(total_pnl, threshold)
                self._circuit_was_tripped = True
            elif not tripped and self._circuit_was_tripped:
                await self._alerter.alert_circuit_breaker_reset(total_pnl)
                self._circuit_was_tripped = False
        except Exception as exc:
            logger.warning("Circuit breaker state check failed: %s", exc)

    async def _store_balance_snapshot(self, usdc: float, pol: float) -> None:
        try:
            from db import queries
            async with self._session_factory() as session:
                async with session.begin():
                    await queries.save_balance_snapshot(session, usdc, pol)
        except Exception as exc:
            logger.warning("Balance snapshot save failed: %s", exc)

    async def _balance_delta_24h(self, current_usdc: float) -> str:
        try:
            from db import queries
            async with self._session_factory() as session:
                prev = await queries.get_balance_snapshot_24h_ago(session)
            if prev is None:
                return ""
            delta = current_usdc - prev
            sign = "+" if delta >= 0 else ""
            return f"  ({sign}${delta:.2f} vs 24h ago)"
        except Exception:
            return ""

    async def _persist(self, statuses: list[dict]) -> None:
        try:
            from db import queries
            async with self._session_factory() as session:
                async with session.begin():
                    for s in statuses:
                        await queries.upsert_component_health(session, s)
        except Exception as exc:
            logger.warning("Health persist failed: %s", exc)
