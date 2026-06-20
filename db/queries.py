import json
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

logger = logging.getLogger(__name__)

from data.models import CryptoPrice, FearGreedData, GammaMarket
from db.models import (
    BalanceSnapshot,
    ComponentHealth,
    Market,
    MarketSnapshot,
    PaperPosition,
    PriceSnapshot,
    ProbabilityEstimate,
    RuntimeConfig,
    SentimentSnapshot,
)
from reasoning.schemas import ProbabilityEstimate as ProbabilityEstimateSchema


# ── DB migration (Phase 5 columns) ───────────────────────────────────────────

async def migrate_db(engine: AsyncEngine) -> None:
    """
    Idempotently add Phase 5 columns to paper_positions if they don't exist.
    SQLite supports ADD COLUMN but not DROP/RENAME, so this is safe to run
    on every startup against an existing database.
    """
    new_cols = [
        ("is_live",       "ALTER TABLE paper_positions ADD COLUMN is_live INTEGER NOT NULL DEFAULT 0"),
        ("order_id",      "ALTER TABLE paper_positions ADD COLUMN order_id TEXT"),
        ("order_status",  "ALTER TABLE paper_positions ADD COLUMN order_status TEXT DEFAULT 'SIMULATED'"),
    ]
    # Phase 7 tables created via Base.metadata.create_all — no ALTER needed
    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(paper_positions)"))
        existing = {row[1] for row in result.fetchall()}
        for col_name, sql in new_cols:
            if col_name not in existing:
                await conn.execute(text(sql))
                logger.info("DB migration: added column '%s' to paper_positions", col_name)


# ── Market / snapshot writes ──────────────────────────────────────────────────

async def upsert_market(session: AsyncSession, market: GammaMarket) -> None:
    now = datetime.now(tz=timezone.utc)
    existing = await session.get(Market, market.id)
    if existing:
        existing.last_updated_at = now
        existing.yes_token_id = market.yes_token_id
        existing.no_token_id = market.no_token_id
    else:
        session.add(
            Market(
                id=market.id,
                question=market.question,
                description=market.description,
                condition_id=market.condition_id,
                yes_token_id=market.yes_token_id,
                no_token_id=market.no_token_id,
                neg_risk=market.neg_risk,
                min_tick=market.min_tick,
                min_order_size=market.min_order_size,
                taker_fee=market.taker_fee,
                end_date=market.end_date,
                event_title=market.event_title,
                first_seen_at=now,
                last_updated_at=now,
            )
        )


async def save_market_snapshot(
    session: AsyncSession, market: GammaMarket, scanned_at: datetime
) -> None:
    session.add(
        MarketSnapshot(
            market_id=market.id,
            yes_price=market.yes_price,
            no_price=market.no_price,
            best_bid=market.best_bid,
            best_ask=market.best_ask,
            spread=market.spread,
            last_trade_price=market.last_trade_price,
            volume=market.volume,
            volume_24hr=market.volume_24hr,
            liquidity=market.liquidity,
            accepting_orders=market.accepting_orders,
            scanned_at=scanned_at,
        )
    )


async def save_price_snapshots(
    session: AsyncSession, prices: list[CryptoPrice]
) -> None:
    for p in prices:
        session.add(
            PriceSnapshot(
                symbol=p.symbol,
                price_usd=p.price_usd,
                change_24h=p.change_24h,
                volume_24h_usd=p.volume_24h_usd,
                scanned_at=p.scanned_at,
            )
        )


async def save_sentiment_snapshot(
    session: AsyncSession, fg: FearGreedData, scanned_at: datetime
) -> None:
    session.add(
        SentimentSnapshot(
            fear_greed_value=fg.value,
            fear_greed_classification=fg.classification,
            scanned_at=scanned_at,
        )
    )


# ── Probability estimate writes ───────────────────────────────────────────────

async def save_probability_estimate(
    session: AsyncSession, est: ProbabilityEstimateSchema
) -> ProbabilityEstimate:
    row = ProbabilityEstimate(
        market_id=est.market_id,
        question=est.question,
        model_probability=est.model_probability,
        implied_probability=est.implied_probability,
        gross_edge=est.gross_edge,
        taker_fee=est.taker_fee,
        net_edge=est.net_edge,
        confidence_low=est.confidence_low,
        confidence_high=est.confidence_high,
        signal=est.signal.value,
        reasoning=est.reasoning,
        key_risks=json.dumps(est.key_risks),
        llm_model=est.llm_model,
        estimated_at=est.estimated_at,
    )
    session.add(row)
    await session.flush()   # populate row.id before returning
    return row


# ── Paper position writes ─────────────────────────────────────────────────────

async def open_position(
    session: AsyncSession,
    market: GammaMarket,
    estimate_row: ProbabilityEstimate,
    direction: str,
    fill_price: float,
    num_shares: float,
    stake_usdc: float,
    is_live: bool = False,
    order_id: str | None = None,
    order_status: str = "SIMULATED",
) -> PaperPosition:
    """Open a position — paper (is_live=False) or live CLOB (is_live=True)."""
    taker_fee = estimate_row.taker_fee or 0.02
    computed_edge = (
        -(estimate_row.net_edge + 2 * taker_fee)
        if direction == "NO"
        else estimate_row.net_edge
    )
    pos = PaperPosition(
        market_id=market.id,
        estimate_id=estimate_row.id,
        question=market.question,
        direction=direction,
        fill_price=fill_price,
        num_shares=num_shares,
        stake_usdc=stake_usdc,
        model_probability=estimate_row.model_probability,
        implied_probability=estimate_row.implied_probability,
        trade_edge=computed_edge,
        status="OPEN",
        opened_at=datetime.now(tz=timezone.utc),
        is_live=is_live,
        order_id=order_id,
        order_status=order_status,
    )
    session.add(pos)
    return pos


async def update_position_order(
    session: AsyncSession,
    position_id: int,
    order_status: str,
    actual_fill_price: float | None = None,
) -> None:
    """Update order status (and optionally correct fill price) after CLOB fill."""
    pos = await session.get(PaperPosition, position_id)
    if pos is None:
        return
    pos.order_status = order_status
    if actual_fill_price is not None and actual_fill_price > 0:
        old_price = pos.fill_price
        pos.fill_price = actual_fill_price
        pos.num_shares = pos.stake_usdc / actual_fill_price
        logger.info(
            "Position %d fill price updated %.4f → %.4f",
            position_id, old_price, actual_fill_price,
        )


async def close_paper_position(
    session: AsyncSession,
    position: PaperPosition,
    outcome: float,
    resolved_at: datetime,
) -> None:
    won = (
        (position.direction == "YES" and outcome == 1.0)
        or (position.direction == "NO" and outcome == 0.0)
    )
    payout = position.num_shares * 1.0 if won else 0.0
    position.outcome = outcome
    position.payout_usdc = payout
    position.pnl_usdc = payout - position.stake_usdc
    position.status = "CLOSED"
    position.closed_at = resolved_at


async def resolve_estimates_for_market(
    session: AsyncSession, market_id: str, outcome: float, resolved_at: datetime
) -> None:
    result = await session.execute(
        select(ProbabilityEstimate)
        .where(ProbabilityEstimate.market_id == market_id)
        .where(ProbabilityEstimate.outcome.is_(None))
    )
    for est in result.scalars().all():
        est.outcome = outcome
        est.model_brier_score = (est.model_probability - outcome) ** 2
        est.market_brier_score = (est.implied_probability - outcome) ** 2
        est.resolved_at = resolved_at


# ── Paper position reads ──────────────────────────────────────────────────────

async def get_open_positions(session: AsyncSession) -> list[PaperPosition]:
    result = await session.execute(
        select(PaperPosition)
        .where(PaperPosition.status == "OPEN")
        .order_by(PaperPosition.opened_at.desc())
    )
    return list(result.scalars().all())


async def get_pending_live_positions(session: AsyncSession) -> list[PaperPosition]:
    """Live positions with PENDING orders — checked by FillMonitor every 5 min."""
    result = await session.execute(
        select(PaperPosition).where(
            PaperPosition.is_live == True,  # noqa: E712
            PaperPosition.order_status == "PENDING",
            PaperPosition.status == "OPEN",
        )
    )
    return list(result.scalars().all())


async def get_open_position_market_ids(session: AsyncSession) -> set[str]:
    result = await session.execute(
        select(PaperPosition.market_id).where(PaperPosition.status == "OPEN")
    )
    return set(result.scalars().all())


# ── Phase 6 — Risk queries ────────────────────────────────────────────────────

async def get_open_position_count(session: AsyncSession) -> int:
    """Count of currently open positions."""
    result = await session.execute(
        select(func.count(PaperPosition.id)).where(PaperPosition.status == "OPEN")
    )
    return result.scalar_one() or 0


async def get_total_open_exposure(session: AsyncSession) -> float:
    """Sum of stake_usdc across all open positions."""
    result = await session.execute(
        select(func.sum(PaperPosition.stake_usdc)).where(PaperPosition.status == "OPEN")
    )
    return float(result.scalar_one() or 0.0)


async def get_total_unrealized_pnl(session: AsyncSession) -> float:
    """
    Approximate unrealized P&L for all open positions using latest market snapshots.
    Used for circuit breaker calculation.
    """
    subq = (
        select(
            MarketSnapshot.market_id,
            func.max(MarketSnapshot.scanned_at).label("latest"),
        )
        .group_by(MarketSnapshot.market_id)
        .subquery()
    )
    result = await session.execute(
        select(PaperPosition, MarketSnapshot.yes_price)
        .outerjoin(subq, PaperPosition.market_id == subq.c.market_id)
        .outerjoin(
            MarketSnapshot,
            and_(
                MarketSnapshot.market_id == subq.c.market_id,
                MarketSnapshot.scanned_at == subq.c.latest,
            ),
        )
        .where(PaperPosition.status == "OPEN")
    )
    total = 0.0
    for pos, yes_price in result.all():
        if yes_price is None:
            yes_price = pos.fill_price
        if pos.direction == "YES":
            current_value = (pos.num_shares or 0) * yes_price
        else:
            current_value = (pos.num_shares or 0) * (1.0 - yes_price)
        total += current_value - (pos.stake_usdc or 0)
    return total


async def get_open_positions_for_underlying(
    session: AsyncSession, keywords: list[str]
) -> int:
    """Count open positions whose question contains any of the given keywords."""
    from sqlalchemy import or_
    if not keywords:
        return 0
    conditions = [PaperPosition.question.ilike(f"%{kw}%") for kw in keywords]
    result = await session.execute(
        select(func.count(PaperPosition.id)).where(
            PaperPosition.status == "OPEN",
            or_(*conditions),
        )
    )
    return result.scalar_one() or 0


async def get_closed_positions(
    session: AsyncSession, limit: int = 50
) -> list[PaperPosition]:
    result = await session.execute(
        select(PaperPosition)
        .where(PaperPosition.status == "CLOSED")
        .order_by(PaperPosition.closed_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


# ── Web UI reads ──────────────────────────────────────────────────────────────

async def get_latest_prices(session: AsyncSession) -> list[PriceSnapshot]:
    subq = (
        select(PriceSnapshot.symbol, func.max(PriceSnapshot.scanned_at).label("latest"))
        .group_by(PriceSnapshot.symbol)
        .subquery()
    )
    result = await session.execute(
        select(PriceSnapshot).join(
            subq,
            and_(
                PriceSnapshot.symbol == subq.c.symbol,
                PriceSnapshot.scanned_at == subq.c.latest,
            ),
        )
    )
    return list(result.scalars().all())


async def get_latest_sentiment(session: AsyncSession) -> SentimentSnapshot | None:
    result = await session.execute(
        select(SentimentSnapshot).order_by(SentimentSnapshot.scanned_at.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def get_active_opportunities(
    session: AsyncSession, hours: int = 24
) -> list[ProbabilityEstimate]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    subq = (
        select(
            ProbabilityEstimate.market_id,
            func.max(ProbabilityEstimate.estimated_at).label("latest"),
        )
        .where(ProbabilityEstimate.estimated_at >= cutoff)
        .group_by(ProbabilityEstimate.market_id)
        .subquery()
    )
    result = await session.execute(
        select(ProbabilityEstimate)
        .join(
            subq,
            and_(
                ProbabilityEstimate.market_id == subq.c.market_id,
                ProbabilityEstimate.estimated_at == subq.c.latest,
            ),
        )
        .where(ProbabilityEstimate.signal != "HOLD")
        .order_by(ProbabilityEstimate.net_edge)
    )
    return list(result.scalars().all())


async def get_markets_with_latest_snapshot(
    session: AsyncSession,
) -> list[tuple[Market, MarketSnapshot]]:
    subq = (
        select(
            MarketSnapshot.market_id,
            func.max(MarketSnapshot.scanned_at).label("latest"),
        )
        .group_by(MarketSnapshot.market_id)
        .subquery()
    )
    result = await session.execute(
        select(Market, MarketSnapshot)
        .join(MarketSnapshot, Market.id == MarketSnapshot.market_id)
        .join(
            subq,
            and_(
                MarketSnapshot.market_id == subq.c.market_id,
                MarketSnapshot.scanned_at == subq.c.latest,
            ),
        )
        .order_by(MarketSnapshot.volume_24hr.desc())
    )
    return result.all()


async def get_calibration_stats(session: AsyncSession) -> dict:
    result = await session.execute(
        select(
            func.count(ProbabilityEstimate.id).label("total_resolved"),
            func.avg(ProbabilityEstimate.model_brier_score).label("avg_model_brier"),
            func.avg(ProbabilityEstimate.market_brier_score).label("avg_market_brier"),
        ).where(ProbabilityEstimate.outcome.is_not(None))
    )
    row = result.one()

    pos_result = await session.execute(
        select(
            func.count(PaperPosition.id).label("total"),
            func.sum(PaperPosition.pnl_usdc).label("net_pnl"),
        ).where(PaperPosition.status == "CLOSED")
    )
    pos_row = pos_result.one()

    wins = await session.execute(
        select(func.count(PaperPosition.id)).where(
            PaperPosition.status == "CLOSED", PaperPosition.pnl_usdc > 0
        )
    )

    total_closed = pos_row.total or 0
    win_count = wins.scalar_one() or 0

    return {
        "total_resolved": row.total_resolved or 0,
        "avg_model_brier": row.avg_model_brier,
        "avg_market_brier": row.avg_market_brier,
        "total_paper_trades": total_closed,
        "win_count": win_count,
        "loss_count": total_closed - win_count,
        "win_rate": (win_count / total_closed * 100) if total_closed else None,
        "net_paper_pnl": pos_row.net_pnl or 0.0,
    }


async def get_recent_estimates(
    session: AsyncSession, limit: int = 50
) -> list[ProbabilityEstimate]:
    result = await session.execute(
        select(ProbabilityEstimate)
        .order_by(ProbabilityEstimate.estimated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_last_scan_time(session: AsyncSession) -> datetime | None:
    result = await session.execute(
        select(func.max(MarketSnapshot.scanned_at))
    )
    return result.scalar_one_or_none()


# ── Runtime config ───────────────────────────────────────────────────────────

async def get_runtime_config(session: AsyncSession) -> dict[str, str]:
    result = await session.execute(select(RuntimeConfig))
    return {row.key: row.value for row in result.scalars().all()}


async def get_all_runtime_configs(session: AsyncSession) -> list[RuntimeConfig]:
    result = await session.execute(
        select(RuntimeConfig).order_by(RuntimeConfig.updated_at.desc())
    )
    return list(result.scalars().all())


async def set_runtime_config(
    session: AsyncSession,
    key: str,
    value: str,
    updated_by: str = "user",
    note: str = "",
) -> None:
    existing = await session.get(RuntimeConfig, key)
    now = datetime.now(tz=timezone.utc)
    if existing:
        existing.value = value
        existing.updated_at = now
        existing.updated_by = updated_by
        existing.note = note
    else:
        session.add(RuntimeConfig(key=key, value=value, updated_at=now, updated_by=updated_by, note=note))


async def get_open_positions_with_prices(session: AsyncSession) -> list[dict]:
    """Open positions augmented with current market price for live unrealised P&L."""
    subq = (
        select(
            MarketSnapshot.market_id,
            func.max(MarketSnapshot.scanned_at).label("latest"),
        )
        .group_by(MarketSnapshot.market_id)
        .subquery()
    )
    result = await session.execute(
        select(PaperPosition, MarketSnapshot.yes_price.label("current_yes"))
        .outerjoin(subq, PaperPosition.market_id == subq.c.market_id)
        .outerjoin(
            MarketSnapshot,
            and_(
                MarketSnapshot.market_id == subq.c.market_id,
                MarketSnapshot.scanned_at == subq.c.latest,
            ),
        )
        .where(PaperPosition.status == "OPEN")
        .order_by(PaperPosition.opened_at.desc())
    )
    rows = []
    for pos, current_yes in result.all():
        yes_price = current_yes if current_yes is not None else pos.fill_price
        if pos.direction == "YES":
            current_value = pos.num_shares * yes_price
        else:
            current_value = pos.num_shares * (1.0 - yes_price)
        unrealized_pnl = current_value - pos.stake_usdc
        # Older positions were stored with net_edge instead of directional trade_edge.
        # Detect and correct: for NO positions, a valid trade_edge is always positive.
        stored_edge = pos.trade_edge or 0.0
        taker_fee = 0.02
        if pos.direction == "NO" and stored_edge <= 0:
            display_edge = -(stored_edge + 2 * taker_fee)
        else:
            display_edge = stored_edge
        rows.append({
            "id": pos.id,
            "direction": pos.direction,
            "fill_price": pos.fill_price,
            "num_shares": pos.num_shares,
            "stake_usdc": pos.stake_usdc,
            "model_probability": pos.model_probability,
            "implied_probability": pos.implied_probability,
            "trade_edge": display_edge,
            "opened_at": pos.opened_at.strftime("%m-%d %H:%M") if pos.opened_at else "",
            "question": pos.question,
            "current_yes": yes_price,
            "current_value": current_value,
            "unrealized_pnl": unrealized_pnl,
            "is_live": bool(pos.is_live),
            "order_id": pos.order_id,
            "order_status": pos.order_status or "SIMULATED",
        })
    return rows


async def auto_adjust_min_edge(
    session: AsyncSession, current_min_edge: float
) -> tuple[float, str | None]:
    """
    Examine the last 15 resolved predictions. If the model consistently
    underperforms vs the market price (Brier model > market + 0.03), tighten
    the min_net_edge threshold. If it significantly outperforms, relax slightly.
    Returns (new_edge, log_message). log_message is None if no change was made.
    """
    result = await session.execute(
        select(
            ProbabilityEstimate.model_brier_score,
            ProbabilityEstimate.market_brier_score,
        )
        .where(ProbabilityEstimate.outcome.is_not(None))
        .where(ProbabilityEstimate.model_brier_score.is_not(None))
        .order_by(ProbabilityEstimate.resolved_at.desc())
        .limit(15)
    )
    rows = result.all()
    if len(rows) < 15:
        status = f"calibrating ({len(rows)}/15 resolved predictions needed)"
        await set_runtime_config(session, "calibration_status", status, "auto-calibration",
                                 f"Need {15 - len(rows)} more resolved predictions")
        return current_min_edge, None

    avg_model = sum(r.model_brier_score for r in rows) / len(rows)
    avg_market = sum(r.market_brier_score for r in rows) / len(rows)

    if avg_model < avg_market - 0.02:
        status_label = "outperforming"
        status_note = f"Model Brier {avg_model:.4f} vs Market {avg_market:.4f} — edge confirmed"
    elif avg_model > avg_market + 0.03:
        status_label = "underperforming"
        status_note = f"Model Brier {avg_model:.4f} vs Market {avg_market:.4f} — tightening"
    else:
        status_label = "on-par"
        status_note = f"Model Brier {avg_model:.4f} vs Market {avg_market:.4f}"

    await set_runtime_config(
        session, "calibration_status", status_label, "auto-calibration", status_note
    )

    new_edge = current_min_edge
    change_msg = None

    if avg_model > avg_market + 0.03:
        new_edge = min(round(current_min_edge + 0.02, 4), 0.25)
        if new_edge != current_min_edge:
            change_msg = (
                f"Increased min_net_edge {current_min_edge:.2f} -> {new_edge:.2f}: "
                f"model Brier {avg_model:.4f} > market {avg_market:.4f}"
            )
    elif avg_model < avg_market - 0.05 and avg_model < 0.15:
        new_edge = max(round(current_min_edge - 0.01, 4), 0.05)
        if new_edge != current_min_edge:
            change_msg = (
                f"Decreased min_net_edge {current_min_edge:.2f} -> {new_edge:.2f}: "
                f"model Brier {avg_model:.4f} << market {avg_market:.4f}"
            )

    return new_edge, change_msg


# ── Legacy reads (keep for scanner + calibration) ────────────────────────────

async def get_open_estimates(session: AsyncSession) -> list[ProbabilityEstimate]:
    result = await session.execute(
        select(ProbabilityEstimate)
        .where(
            ProbabilityEstimate.signal != "HOLD",
            ProbabilityEstimate.outcome.is_(None),
        )
        .order_by(ProbabilityEstimate.estimated_at.desc())
    )
    return list(result.scalars().all())


async def get_recent_snapshots(
    session: AsyncSession, market_id: str, limit: int = 48
) -> list[MarketSnapshot]:
    result = await session.execute(
        select(MarketSnapshot)
        .where(MarketSnapshot.market_id == market_id)
        .order_by(MarketSnapshot.scanned_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_known_market_ids(session: AsyncSession) -> set[str]:
    result = await session.execute(select(Market.id))
    return set(result.scalars().all())


# ── Phase 7 — Health + balance queries ───────────────────────────────────────

async def get_recent_estimate_count(session: AsyncSession, hours: int = 24) -> int:
    """Count of LLM estimates generated in the last N hours."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    result = await session.execute(
        select(func.count(ProbabilityEstimate.id))
        .where(ProbabilityEstimate.estimated_at >= cutoff)
    )
    return result.scalar_one() or 0


async def upsert_component_health(session: AsyncSession, status: dict) -> None:
    """Upsert one health status row (one row per component name)."""
    existing = await session.get(ComponentHealth, status["name"])
    now = datetime.now(tz=timezone.utc)
    if existing:
        existing.display_name = status["display_name"]
        existing.category = status["category"]
        existing.status = status["status"]
        existing.detail = status["detail"]
        existing.required = status.get("required", False)
        existing.latency_ms = status.get("latency_ms")
        existing.checked_at = now
    else:
        session.add(ComponentHealth(
            component=status["name"],
            display_name=status["display_name"],
            category=status["category"],
            status=status["status"],
            detail=status["detail"],
            required=status.get("required", False),
            latency_ms=status.get("latency_ms"),
            checked_at=now,
        ))


async def get_component_health(session: AsyncSession) -> list[ComponentHealth]:
    """All component health rows, ordered by category then name."""
    result = await session.execute(
        select(ComponentHealth).order_by(ComponentHealth.category, ComponentHealth.component)
    )
    return list(result.scalars().all())


async def save_balance_snapshot(
    session: AsyncSession, usdc: float, pol: float
) -> None:
    session.add(BalanceSnapshot(
        usdc_balance=usdc,
        pol_balance=pol,
        recorded_at=datetime.now(tz=timezone.utc),
    ))


async def get_balance_snapshot_24h_ago(session: AsyncSession) -> float | None:
    """USDC balance closest to 24 hours ago, for delta display."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    result = await session.execute(
        select(BalanceSnapshot.usdc_balance)
        .where(BalanceSnapshot.recorded_at <= cutoff)
        .order_by(BalanceSnapshot.recorded_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_balance_history(
    session: AsyncSession, hours: int = 48
) -> list[BalanceSnapshot]:
    """Recent balance snapshots for charting."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
    result = await session.execute(
        select(BalanceSnapshot)
        .where(BalanceSnapshot.recorded_at >= cutoff)
        .order_by(BalanceSnapshot.recorded_at.asc())
    )
    return list(result.scalars().all())
