"""
Web UI routes. Mounted onto the main FastAPI app in server.py.
All routes require HTTP Basic Auth.
"""
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from data.gamma import _is_non_crypto_contamination
from db import queries

_settings = get_settings()
_security = HTTPBasic()
_templates = Jinja2Templates(directory="web/templates")

router = APIRouter()

# ── Editable parameters (strategy + risk) ────────────────────────────────────
# Each entry: key → (label, type, env_default, html_step, description, min, max)
_EDITABLE_PARAMS: dict[str, tuple] = {
    # Strategy
    "min_net_edge": (
        "Min Edge Threshold", float, 0.10, "0.01",
        "Minimum model edge required to flag a BUY signal (0.05–0.25)",
        0.05, 0.25,
    ),
    "max_markets_per_scan": (
        "Max Markets / Scan", int, 10, "1",
        "Number of top-volume markets sent to the LLM per scan",
        1, 50,
    ),
    "paper_position_size_usd": (
        "Base Position Size ($)", float, 10.0, "1",
        "USDC staked per trade when Kelly sizing is disabled",
        1.0, 1000.0,
    ),
    "min_market_volume": (
        "Min Market Volume ($)", float, 10_000.0, "1000",
        "Markets below this lifetime volume are excluded",
        0.0, None,
    ),
    "scan_interval_minutes": (
        "Scan Interval (min)", int, 15, "1",
        "How often the scanner runs — requires restart to take effect",
        1, 1440,
    ),
    # Risk — Phase 6
    "drawdown_limit_usd": (
        "Circuit Breaker ($)", float, 50.0, "5",
        "Pause new trades if total P&L (realized + unrealized) drops below this. 0 = disabled.",
        0.0, None,
    ),
    "max_open_positions": (
        "Max Open Positions", int, 10, "1",
        "Maximum number of concurrent open positions across the portfolio",
        1, 100,
    ),
    "max_exposure_usd": (
        "Max Exposure ($)", float, 200.0, "10",
        "Maximum total USDC staked across all open positions. 0 = no limit.",
        0.0, None,
    ),
    "max_position_size_usd": (
        "Max Position Size ($)", float, 50.0, "1",
        "Hard per-trade cap regardless of Kelly output",
        1.0, None,
    ),
    "kelly_fraction": (
        "Kelly Fraction", float, 0.25, "0.01",
        "Fraction of full Kelly to use (0.25 = quarter-Kelly). Set 0 to use Base Position Size instead.",
        0.0, 1.0,
    ),
}

_STRATEGY_KEYS = {"min_net_edge", "max_markets_per_scan", "paper_position_size_usd",
                  "min_market_volume", "scan_interval_minutes"}
_RISK_KEYS = {"drawdown_limit_usd", "max_open_positions", "max_exposure_usd",
              "max_position_size_usd", "kelly_fraction"}


# ── Auth ──────────────────────────────────────────────────────────────────────

def _require_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(_security)],
) -> str:
    ok = secrets.compare_digest(
        credentials.username.encode(), _settings.web_username.encode()
    ) and secrets.compare_digest(
        credentials.password.encode(), _settings.web_password.encode()
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            detail="Unauthorised",
        )
    return credentials.username


# ── DB session dependency ─────────────────────────────────────────────────────

async def _get_session() -> AsyncSession:
    from server import get_session_factory
    factory = get_session_factory()
    async with factory() as session:
        yield session


Auth = Annotated[str, Depends(_require_auth)]
DB = Annotated[AsyncSession, Depends(_get_session)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trading_context() -> dict:
    from server import get_executor, get_wallet_state, get_health_monitor
    executor = get_executor()
    hm = get_health_monitor()
    halted = hm.is_trading_halted()
    return {
        "trading_mode": executor.mode_label,
        "is_live": executor.is_live,
        "wallet_address_masked": executor.wallet_address_masked,
        "wallet_state": get_wallet_state(),
        "trading_halted": halted,
        "halt_reasons": hm.get_halt_reasons() if halted else [],
    }


def _clamp(value, lo, hi):
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


# ── Page routes ───────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _: Auth, session: DB):
    from server import get_risk_engine, get_health_monitor
    prices = await queries.get_latest_prices(session)
    sentiment = await queries.get_latest_sentiment(session)
    raw_opps = await queries.get_active_opportunities(session)
    opportunities = [o for o in raw_opps if not _is_non_crypto_contamination(o.question)]
    open_positions = await queries.get_open_positions_with_prices(session)
    stats = await queries.get_calibration_stats(session)
    last_scan = await queries.get_last_scan_time(session)
    risk_status = await get_risk_engine().get_status(session)
    health_statuses = get_health_monitor().get_statuses()

    return _templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "prices": prices,
            "sentiment": sentiment,
            "opportunities": opportunities,
            "open_positions": open_positions,
            "stats": stats,
            "last_scan": last_scan,
            "risk_status": risk_status,
            "health_statuses": health_statuses,
            "settings": _settings,
            **_trading_context(),
        },
    )


@router.get("/markets", response_class=HTMLResponse)
async def markets(request: Request, _: Auth, session: DB):
    rows = await queries.get_markets_with_latest_snapshot(session)
    return _templates.TemplateResponse(
        "markets.html",
        {"request": request, "rows": rows, **_trading_context()},
    )


@router.get("/positions", response_class=HTMLResponse)
async def positions(request: Request, _: Auth, session: DB):
    open_pos = await queries.get_open_positions(session)
    closed_pos = await queries.get_closed_positions(session)
    stats = await queries.get_calibration_stats(session)
    return _templates.TemplateResponse(
        "positions.html",
        {
            "request": request,
            "open_positions": open_pos,
            "closed_positions": closed_pos,
            "stats": stats,
            **_trading_context(),
        },
    )


@router.get("/calibration", response_class=HTMLResponse)
async def calibration(request: Request, _: Auth, session: DB):
    stats = await queries.get_calibration_stats(session)
    recent = await queries.get_recent_estimates(session, limit=50)
    return _templates.TemplateResponse(
        "calibration.html",
        {"request": request, "stats": stats, "estimates": recent, **_trading_context()},
    )


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request, _: Auth, session: DB):
    from server import get_risk_engine
    stats = await queries.get_calibration_stats(session)
    overrides = await queries.get_runtime_config(session)
    config_rows = await queries.get_all_runtime_configs(session)
    risk_status = await get_risk_engine().get_status(session)

    effective = {}
    for key, (label, typ, default, step, desc, *bounds) in _EDITABLE_PARAMS.items():
        raw = overrides.get(key)
        effective[key] = typ(raw) if raw is not None else default

    return _templates.TemplateResponse(
        "configs.html",
        {
            "request": request,
            "stats": stats,
            "effective": effective,
            "config_rows": config_rows,
            "editable_params": _EDITABLE_PARAMS,
            "strategy_keys": _STRATEGY_KEYS,
            "risk_keys": _RISK_KEYS,
            "settings": _settings,
            "overrides": overrides,
            "risk_status": risk_status,
            **_trading_context(),
        },
    )


@router.post("/config", response_class=HTMLResponse)
async def config_save(request: Request, _: Auth, session: DB):
    form = await request.form()
    async with session.begin():
        for key, (label, typ, default, step, desc, *bounds) in _EDITABLE_PARAMS.items():
            raw = form.get(key)
            if raw is None:
                continue
            try:
                val = typ(raw)
                lo = bounds[0] if len(bounds) > 0 else None
                hi = bounds[1] if len(bounds) > 1 else None
                val = _clamp(val, lo, hi)
                await queries.set_runtime_config(
                    session, key, str(val), "user", "Manual edit via web UI"
                )
            except (ValueError, TypeError):
                pass
    return RedirectResponse("/config", status_code=303)


# ── JSON API endpoints ────────────────────────────────────────────────────────

@router.get("/api/logs", include_in_schema=False)
async def api_logs(_: Auth):
    from server import get_log_buffer
    return JSONResponse(get_log_buffer()[:100])


@router.get("/api/positions/live", include_in_schema=False)
async def api_positions_live(_: Auth, session: DB):
    rows = await queries.get_open_positions_with_prices(session)
    return JSONResponse(rows)


@router.get("/api/prices/live", include_in_schema=False)
async def api_prices_live(_: Auth, session: DB):
    prices = await queries.get_latest_prices(session)
    return JSONResponse([
        {"symbol": p.symbol, "price_usd": p.price_usd, "change_24h": p.change_24h}
        for p in sorted(prices, key=lambda x: x.price_usd, reverse=True)
    ])


@router.get("/api/wallet", include_in_schema=False)
async def api_wallet(_: Auth):
    from server import refresh_wallet_state
    state = await refresh_wallet_state()
    if state is None:
        return JSONResponse({"mode": "PAPER", "connected": False})
    return JSONResponse({
        "mode": "LIVE",
        "connected": True,
        "address_masked": state["address_masked"],
        "usdc_balance": state["usdc_balance"],
        "pol_balance": state["pol_balance"],
        "clob_connected": state["clob_connected"],
    })


@router.get("/api/risk", include_in_schema=False)
async def api_risk(_: Auth, session: DB):
    from server import get_risk_engine
    status = await get_risk_engine().get_status(session)
    return JSONResponse(status)


@router.get("/api/health", include_in_schema=False)
async def api_health(_: Auth):
    """System health snapshot polled by dashboard JS every 30s."""
    from server import get_health_monitor
    hm = get_health_monitor()
    return JSONResponse({
        "halted": hm.is_trading_halted(),
        "halt_reasons": hm.get_halt_reasons(),
        "statuses": hm.get_statuses(),
    })
