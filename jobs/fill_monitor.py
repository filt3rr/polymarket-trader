"""
Fill monitor job — tracks pending live CLOB orders and updates their status.

Runs every 5 minutes (configured in server.py).
Only active when PAPER_TRADING=False and a ClobExecutor is available.

Lifecycle of a live position:
  PENDING  → order placed, waiting for fill
  FILLED   → order matched, position is active
  CANCELLED → order timed out or was rejected; position removed from open set
"""
import asyncio
import logging
from datetime import datetime, timezone

from db import queries

logger = logging.getLogger(__name__)


class FillMonitor:
    def __init__(self, session_factory, clob_executor, timeout_minutes: int = 10) -> None:
        self._session_factory = session_factory
        self._clob = clob_executor          # ClobExecutor | None
        self._timeout_minutes = timeout_minutes

    async def run(self) -> None:
        if self._clob is None:
            return

        async with self._session_factory() as session:
            pending = await queries.get_pending_live_positions(session)

        if not pending:
            return

        logger.info("Fill monitor: checking %d pending order(s)", len(pending))
        now = datetime.now(tz=timezone.utc)

        for position in pending:
            if not position.order_id:
                continue
            try:
                await self._check_position(position, now)
            except Exception as exc:
                logger.error(
                    "Fill check error for position %d (order %.12s): %s",
                    position.id, position.order_id, exc,
                )

    async def _check_position(self, position, now: datetime) -> None:
        order_data = await asyncio.to_thread(
            self._clob.get_order_status, position.order_id
        )
        raw_status = (order_data.get("status") or "").lower()

        # ── Fully filled ──────────────────────────────────────────────────────
        if raw_status in ("matched", "filled"):
            avg_price = await asyncio.to_thread(
                self._clob.get_average_fill_price, position.order_id
            )
            async with self._session_factory() as session:
                async with session.begin():
                    await queries.update_position_order(
                        session,
                        position_id=position.id,
                        order_status="FILLED",
                        actual_fill_price=avg_price,
                    )
            logger.info(
                "Order FILLED: position %d  order=%.12s  avg_price=%.4f",
                position.id, position.order_id, avg_price or position.fill_price,
            )
            return

        # ── Already cancelled on exchange ─────────────────────────────────────
        if raw_status in ("cancelled", "canceled"):
            async with self._session_factory() as session:
                async with session.begin():
                    await queries.update_position_order(
                        session,
                        position_id=position.id,
                        order_status="CANCELLED",
                    )
            logger.warning(
                "Order CANCELLED on exchange: position %d  order=%.12s",
                position.id, position.order_id,
            )
            return

        # ── Still live — check timeout ────────────────────────────────────────
        if position.opened_at:
            age_minutes = (now - position.opened_at).total_seconds() / 60
            if age_minutes > self._timeout_minutes:
                cancelled = await asyncio.to_thread(
                    self._clob.cancel_order, position.order_id
                )
                new_status = "CANCELLED" if cancelled else "PENDING"
                async with self._session_factory() as session:
                    async with session.begin():
                        await queries.update_position_order(
                            session,
                            position_id=position.id,
                            order_status=new_status,
                        )
                logger.warning(
                    "Order timed out (%.0f min): position %d  order=%.12s  cancelled=%s",
                    age_minutes, position.id, position.order_id, cancelled,
                )
