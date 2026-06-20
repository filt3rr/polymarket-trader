"""
Resolution checker job.

Runs on a schedule. For every open paper position, fetches the current
market state from Gamma and closes the position if the market has resolved.
Also back-fills Brier scores on ProbabilityEstimate rows for the same market.
"""
import json
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db import queries
from db.models import PaperPosition

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


async def _fetch_market(client: httpx.AsyncClient, market_id: str) -> dict | None:
    try:
        resp = await client.get(f"/markets/{market_id}", timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data[0] if isinstance(data, list) else data
    except Exception as exc:
        logger.warning("Failed to fetch market %s: %s", market_id, exc)
        return None


def _parse_outcome(market_data: dict) -> float | None:
    """
    Returns 1.0 if YES resolved, 0.0 if NO resolved, None if still open.
    Resolution is detected when the market is closed and one outcome price
    has converged to 1.0 (winner) and the other to 0.0 (loser).
    """
    if not market_data.get("closed", False):
        return None

    prices_raw = market_data.get("outcomePrices", '["0.5","0.5"]')
    try:
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        yes_price = float(prices[0])
    except Exception:
        return None

    if yes_price >= 0.99:
        return 1.0   # YES resolved
    if yes_price <= 0.01:
        return 0.0   # NO resolved
    return None      # Market closed but outcome not yet determined


class ResolutionChecker:
    def __init__(self, session_factory: async_sessionmaker, alerter=None) -> None:
        self._session_factory = session_factory
        self._alerter = alerter  # TelegramAlerter | None

    async def run(self) -> None:
        async with self._session_factory() as session:
            open_positions: list[PaperPosition] = await queries.get_open_positions(session)

        if not open_positions:
            logger.debug("Resolution check: no open positions.")
            return

        logger.info("Resolution check: %d open positions to check.", len(open_positions))
        resolved_count = 0

        async with httpx.AsyncClient(base_url=GAMMA_BASE) as client:
            for position in open_positions:
                market_data = await _fetch_market(client, position.market_id)
                if market_data is None:
                    continue

                outcome = _parse_outcome(market_data)
                if outcome is None:
                    continue

                resolved_at = datetime.now(tz=timezone.utc)
                async with self._session_factory() as session:
                    async with session.begin():
                        pos = await session.get(PaperPosition, position.id)
                        if pos is None or pos.status != "OPEN":
                            continue
                        await queries.close_paper_position(session, pos, outcome, resolved_at)
                        await queries.resolve_estimates_for_market(
                            session, position.market_id, outcome, resolved_at
                        )

                won = (
                    (position.direction == "YES" and outcome == 1.0)
                    or (position.direction == "NO" and outcome == 0.0)
                )
                pnl = (position.num_shares if won else 0.0) - position.stake_usdc
                result_label = "WIN" if won else "LOSS"
                outcome_str = "YES" if outcome == 1.0 else "NO"
                logger.info(
                    "Resolved %s [%s direction=%s outcome=%s pnl=$%.2f]",
                    position.question[:60],
                    result_label,
                    position.direction,
                    outcome_str,
                    pnl,
                )
                if self._alerter:
                    await self._alerter.alert_position_resolved(
                        direction=position.direction,
                        question=position.question,
                        pnl=pnl,
                        outcome_str=outcome_str,
                    )
                resolved_count += 1

        logger.info("Resolution check complete: %d positions resolved.", resolved_count)

        # Run auto-calibration adjustment after any resolutions
        if resolved_count > 0:
            async with self._session_factory() as session:
                overrides = await queries.get_runtime_config(session)
                current_edge = float(overrides.get("min_net_edge", 0.10))
                new_edge, change_msg = await queries.auto_adjust_min_edge(session, current_edge)
                async with session.begin():
                    if change_msg:
                        await queries.set_runtime_config(
                            session, "min_net_edge", str(new_edge),
                            "auto-calibration", change_msg,
                        )
                        logger.info("Auto-calibration: %s", change_msg)
                    await queries.set_runtime_config(
                        session, "last_calibration_check",
                        datetime.now(tz=timezone.utc).isoformat(),
                        "auto-calibration", f"Ran after {resolved_count} resolutions",
                    )
