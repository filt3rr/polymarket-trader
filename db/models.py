from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Market(Base):
    """One row per unique Polymarket market — upserted on each scan."""

    __tablename__ = "markets"

    id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    description = Column(Text)
    condition_id = Column(String)
    yes_token_id = Column(String)
    no_token_id = Column(String)
    neg_risk = Column(Boolean, default=False)
    min_tick = Column(Float, default=0.01)
    min_order_size = Column(Float, default=5.0)
    taker_fee = Column(Float, default=0.05)
    end_date = Column(DateTime(timezone=True))
    event_title = Column(String)
    first_seen_at = Column(DateTime(timezone=True), default=_now)
    last_updated_at = Column(DateTime(timezone=True))


class MarketSnapshot(Base):
    """Time-series price and volume data per market, captured on each scan."""

    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    yes_price = Column(Float)
    no_price = Column(Float)
    best_bid = Column(Float)
    best_ask = Column(Float)
    spread = Column(Float)
    last_trade_price = Column(Float)
    volume = Column(Float)
    volume_24hr = Column(Float)
    liquidity = Column(Float)
    accepting_orders = Column(Boolean)
    scanned_at = Column(DateTime(timezone=True), nullable=False)


class PriceSnapshot(Base):
    """BTC/ETH/etc. spot prices captured on each scan."""

    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    price_usd = Column(Float)
    change_24h = Column(Float)
    volume_24h_usd = Column(Float)
    scanned_at = Column(DateTime(timezone=True), nullable=False)


class SentimentSnapshot(Base):
    """Crypto Fear & Greed Index captured on each scan."""

    __tablename__ = "sentiment_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fear_greed_value = Column(Integer)
    fear_greed_classification = Column(String)
    scanned_at = Column(DateTime(timezone=True), nullable=False)


class ProbabilityEstimate(Base):
    """
    LLM-generated probability estimate for a market at a point in time.
    Brier score columns are populated when the market resolves.
    """

    __tablename__ = "probability_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)

    model_probability = Column(Float, nullable=False)
    implied_probability = Column(Float, nullable=False)
    gross_edge = Column(Float, nullable=False)
    taker_fee = Column(Float, nullable=False)
    net_edge = Column(Float, nullable=False)

    confidence_low = Column(Float)
    confidence_high = Column(Float)

    signal = Column(String, nullable=False)   # BUY_YES | BUY_NO | HOLD
    reasoning = Column(Text)
    key_risks = Column(Text)                  # JSON-encoded list[str]

    llm_model = Column(String)
    estimated_at = Column(DateTime(timezone=True), nullable=False)

    # Populated on resolution
    outcome = Column(Float)                   # 1.0 = YES, 0.0 = NO
    model_brier_score = Column(Float)         # (model_prob - outcome)^2
    market_brier_score = Column(Float)        # (implied_prob - outcome)^2
    resolved_at = Column(DateTime(timezone=True))


class RuntimeConfig(Base):
    """Key-value store for runtime-editable configuration overrides."""

    __tablename__ = "runtime_config"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    updated_by = Column(String, default="user")  # "user" | "auto-calibration"
    note = Column(Text, default="")


class PaperPosition(Base):
    """
    One row per opened position — paper or live.
    Paper positions: is_live=False, order_id=None, order_status="SIMULATED".
    Live positions:  is_live=True,  order_id=CLOB order ID, order_status tracks fill lifecycle.
    """

    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, ForeignKey("markets.id", ondelete="CASCADE"), nullable=False)
    estimate_id = Column(Integer, ForeignKey("probability_estimates.id"), nullable=True)
    question = Column(Text, nullable=False)

    direction = Column(String, nullable=False)   # YES | NO
    fill_price = Column(Float, nullable=False)   # per-share entry price
    num_shares = Column(Float, nullable=False)
    stake_usdc = Column(Float, nullable=False)   # fill_price * num_shares ≈ position_size

    # Snapshot of model state at entry
    model_probability = Column(Float)
    implied_probability = Column(Float)
    trade_edge = Column(Float)

    status = Column(String, nullable=False, default="OPEN")  # OPEN | CLOSED
    opened_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    # Phase 5 — live execution tracking
    is_live = Column(Boolean, default=False, nullable=False)
    order_id = Column(String, nullable=True)     # Polymarket CLOB order ID
    order_status = Column(String, default="SIMULATED")  # SIMULATED|PENDING|FILLED|CANCELLED

    # Populated on resolution
    outcome = Column(Float)          # 1.0 = YES resolves, 0.0 = NO resolves
    payout_usdc = Column(Float)      # num_shares * 1.0 if won, else 0.0
    pnl_usdc = Column(Float)         # payout_usdc - stake_usdc
    closed_at = Column(DateTime(timezone=True))


class ComponentHealth(Base):
    """
    Phase 7 — Latest health status per system component.
    One row per component, upserted on every health check run.
    """

    __tablename__ = "component_health"

    component = Column(String, primary_key=True)   # e.g. "database", "gamma_api"
    display_name = Column(String, nullable=False)
    category = Column(String, nullable=False)      # required | data | jobs | model | live
    status = Column(String, nullable=False)        # UP | DOWN | DEGRADED | UNKNOWN | N/A
    detail = Column(Text)
    required = Column(Boolean, default=False)
    latency_ms = Column(Float)
    checked_at = Column(DateTime(timezone=True), nullable=False)


class BalanceSnapshot(Base):
    """
    Phase 7 — On-chain wallet balance history (live mode only).
    Recorded on each health check when wallet is connected.
    """

    __tablename__ = "balance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    usdc_balance = Column(Float, nullable=False)
    pol_balance = Column(Float, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=_now)
