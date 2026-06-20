"""
Combined entry point: FastAPI web server + APScheduler background jobs.
Run with: uvicorn server:app --host 0.0.0.0 --port 8000
Or:        python server.py
"""
import asyncio
import logging
import os
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from config import get_settings
from scanner import MarketScanner
from jobs.resolution import ResolutionChecker

_settings = get_settings()

# ── In-memory log ring buffer ─────────────────────────────────────────────────

_log_buffer: deque = deque(maxlen=300)

_SKIP_SOURCES = frozenset({
    "httpx", "httpcore", "sqlalchemy", "apscheduler",
    "multipart", "uvicorn", "fastapi", "asyncio",
})


class _BufferHandler(logging.Handler):
    """Captures log records into the in-memory ring buffer for the dashboard."""

    def emit(self, record: logging.LogRecord) -> None:
        root = record.name.split(".")[0]
        if root in _SKIP_SOURCES:
            return
        _log_buffer.appendleft({
            "t": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S"),
            "level": record.levelname,
            "src": record.name.split(".")[-1],
            "msg": record.getMessage(),
        })


def get_log_buffer() -> list[dict]:
    return list(_log_buffer)


# ── Logging setup ─────────────────────────────────────────────────────────────

_buf_handler = _BufferHandler()
_buf_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=_settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().addHandler(_buf_handler)

for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "apscheduler", "multipart"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_scanner: MarketScanner | None = None
_scheduler: AsyncIOScheduler | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _scanner, _scheduler

    _scanner = MarketScanner(_settings)
    await _scanner.init_db()

    resolution_checker = ResolutionChecker(_scanner._session_factory)

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _scanner.scan,
        trigger=IntervalTrigger(minutes=_settings.scan_interval_minutes),
        id="market_scan",
        name="Market Scan",
        max_instances=1,
        misfire_grace_time=60,
    )
    _scheduler.add_job(
        resolution_checker.run,
        trigger=IntervalTrigger(minutes=_settings.resolution_check_interval_minutes),
        id="resolution_check",
        name="Resolution Checker",
        max_instances=1,
        misfire_grace_time=60,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started — scan every %dmin, resolution check every %dmin.",
        _settings.scan_interval_minutes,
        _settings.resolution_check_interval_minutes,
    )

    # Run initial scan in background so the web server starts immediately
    asyncio.create_task(_scanner.scan())

    yield

    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


# Build the FastAPI app and attach web routes
app = FastAPI(title="Polymarket Trader", lifespan=_lifespan)

from web.app import router as web_router  # noqa: E402
app.include_router(web_router)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "paper_trading": _settings.paper_trading}


def get_session_factory():
    """Dependency accessor for web routes."""
    if _scanner is None:
        raise RuntimeError("Scanner not initialised")
    return _scanner._session_factory


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
