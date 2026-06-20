"""
Web UI routes. Mounted onto the main FastAPI app in server.py.
All routes require HTTP Basic Auth.
"""
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db import queries

_settings = get_settings()
_security = HTTPBasic()
_templates = Jinja2Templates(directory="web/templates")

router = APIRouter()

# Editable strategy parameters
_EDITABLE_PARAMS = {
    "min_net_edge": ("Min Edge Threshold", float, 0.10, "0.01",
                     "Minimum model edge required to flag a BUY signal (0.05–0.25)"),
    "max_markets_per_scan": ("Max Markets / Scan", int, 10, "1",
                             "Number of top-volume markets sent to the LLM per scan"),
    "paper_position_size_usd": ("Position Size ($)", float, 10.0, "1",
                                "USDC staked per trade (paper or live)"),
    "min_market_volume": ("Min Market Volume ($)", float, 10_000.0, "1000",
                          "Markets below this lifetime volume are excluded"),
    "scan_interval_minutes": ("Scan Interval (min)", int, 15, "1",
                              "How often the scanner runs — requires restart to take effect"),
}


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
    """Injects trading mode + wallet info into every template context."""
    from server import get_executor, get_wallet_state
    executor = get_executor()
    return {
        "trading_mode": executor.mode_label,
        "is_live": executor.is_live,
        "wallet_address_masked": executor.wallet_address_masked,
        "wallet_state": get_wallet_state(),
    }


# ── Page routes ───────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _: Auth, session: DB):
    prices = await queries.get_latest_prices(session)
    sentiment = await queries.get_latest_sentiment(session)
    opportunities = await queries.get_active_opportunities(session)
    open_positions = await queries.get_open_positions_with_prices(session)
    stats = await queries.get_calibration_stats(session)
    last_scan = await queries.get_last_scan_time(session)

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
    stats = await queries.get_calibration_stats(session)
    overrides = await queries.get_runtime_config(session)
    config_rows = await queries.get_all_runtime_configs(session)

    effective = {}
    for key, (label, typ, default, step, desc) in _EDITABLE_PARAMS.items():
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
            "settings": _settings,
            "overrides": overrides,
            **_trading_context(),
        },
    )


@router.post("/config", response_class=HTMLResponse)
async def config_save(
    request: Request,
    _: Auth,
    session: DB,
    min_net_edge: float = Form(...),
    max_markets_per_scan: int = Form(...),
    paper_position_size_usd: float = Form(...),
    min_market_volume: float = Form(...),
    scan_interval_minutes: int = Form(...),
):
    updates = {
        "min_net_edge": (max(0.05, min(0.25, min_net_edge)), float),
        "max_markets_per_scan": (max(1, min(50, max_markets_per_scan)), int),
        "paper_position_size_usd": (max(1.0, min(1000.0, paper_position_size_usd)), float),
        "min_market_volume": (max(0.0, min_market_volume), float),
        "scan_interval_minutes": (max(1, min(1440, scan_interval_minutes)), int),
    }
    async with session.begin():
        for key, (val, _) in updates.items():
            await queries.set_runtime_config(session, key, str(val), "user", "Manual edit")

    return RedirectResponse("/config", status_code=303)


# ── JSON API endpoints (polled by dashboard JS) ───────────────────────────────

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
        {
            "symbol": p.symbol,
            "price_usd": p.price_usd,
            "change_24h": p.change_24h,
        }
        for p in sorted(prices, key=lambda x: x.price_usd, reverse=True)
    ])


@router.get("/api/wallet", include_in_schema=False)
async def api_wallet(_: Auth):
    """Live wallet balance — polled by config page JS in live mode."""
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
