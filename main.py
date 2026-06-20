import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import get_settings
from scanner import MarketScanner


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


async def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Polymarket Trader starting — PAPER_TRADING=%s", settings.paper_trading)

    scanner = MarketScanner(settings)
    await scanner.init_db()

    logger.info("Running initial scan...")
    await scanner.scan()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scanner.scan,
        trigger=IntervalTrigger(minutes=settings.scan_interval_minutes),
        id="market_scan",
        name="Crypto Market Scan",
        max_instances=1,
        misfire_grace_time=60,
    )
    scheduler.start()
    logger.info(
        "Scheduler running — scanning every %d min. Press Ctrl+C to stop.",
        settings.scan_interval_minutes,
    )

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
