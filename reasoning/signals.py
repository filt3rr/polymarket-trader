"""
Quantitative signal extraction for a prediction market.

Builds a structured MarketSignals object from a GammaMarket + available
price/sentiment data. This feeds directly into the LLM prompt — no raw
API responses are passed to the model.
"""
import re
from dataclasses import dataclass, field
from data.models import CryptoPrice, FearGreedData, GammaMarket

# Maps keywords in a question to the relevant tracked symbol
_COIN_KEYWORDS: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "xrp": "XRP",
    "ripple": "XRP",
    "bnb": "BNB",
    "doge": "DOGE",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
    "chainlink": "LINK",
    "link": "LINK",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "polygon": "POL",
    "pol": "POL",
}

# Phrases that indicate an all-time high question (no dollar amount in the question)
_ATH_PHRASES = ("all time high", "all-time high", "ath", "new high", "record high")


@dataclass
class PriceTarget:
    """Extracted price target from a market question."""
    symbol: str
    target_usd: float
    current_usd: float
    gain_needed_pct: float       # (target / current - 1) * 100
    multiple_needed: float       # target / current


@dataclass
class AthContext:
    """All-time high data for the primary coin in this market."""
    symbol: str
    ath_usd: float
    current_usd: float
    pct_from_ath: float          # negative = below ATH (e.g. -13.5 means 13.5% below)
    gain_to_ath_pct: float       # gain needed to reach ATH (positive)
    is_ath_question: bool        # True when the market question explicitly asks about ATH


@dataclass
class MarketSignals:
    """All signals available for one market at scan time."""

    # Identity
    market_id: str
    question: str
    event_title: str

    # Market structure
    implied_prob: float          # market's current YES price
    days_to_resolution: float | None
    volume_24hr: float
    total_volume: float
    liquidity: float
    spread: float | None
    taker_fee: float
    break_even_edge: float       # == taker_fee (minimum gross edge needed)

    # Spot prices (all tracked coins at scan time)
    spot_prices: dict[str, float] = field(default_factory=dict)
    price_changes_24h: dict[str, float] = field(default_factory=dict)
    price_changes_7d: dict[str, float] = field(default_factory=dict)
    market_caps: dict[str, float] = field(default_factory=dict)

    # Macro sentiment
    fear_greed_value: int | None = None
    fear_greed_classification: str | None = None

    # Price target (populated when question references a dollar price level)
    price_target: PriceTarget | None = None

    # ATH context (populated when the primary coin is identifiable)
    ath_context: AthContext | None = None

    # 24h range for the primary coin
    coin_high_24h: float | None = None
    coin_low_24h: float | None = None

    # Derived momentum: 24h vol as % of average daily vol over market lifetime
    volume_momentum: float | None = None


def build_signals(
    market: GammaMarket,
    prices: list[CryptoPrice],
    fear_greed: FearGreedData | None,
) -> MarketSignals:
    price_map = {p.symbol: p.price_usd for p in prices}
    change_24h_map = {p.symbol: p.change_24h for p in prices}
    change_7d_map = {p.symbol: p.change_7d for p in prices if p.change_7d is not None}
    market_cap_map = {p.symbol: p.market_cap_usd for p in prices if p.market_cap_usd is not None}

    days = market.days_to_resolution

    volume_momentum: float | None = None
    if days is not None and market.volume > 0 and days > 0:
        avg_daily = market.volume / max(days, 1)
        if avg_daily > 0:
            volume_momentum = market.volume_24hr / avg_daily

    # Identify the primary coin referenced by this market question
    q_lower = market.question.lower()
    primary_symbol: str | None = None
    for kw, sym in _COIN_KEYWORDS.items():
        if kw in q_lower:
            primary_symbol = sym
            break

    # ATH context and 24h range for the primary coin
    ath_context: AthContext | None = None
    coin_high_24h: float | None = None
    coin_low_24h: float | None = None

    if primary_symbol:
        primary_price_obj = next((p for p in prices if p.symbol == primary_symbol), None)
        if primary_price_obj:
            coin_high_24h = primary_price_obj.high_24h
            coin_low_24h = primary_price_obj.low_24h
            if primary_price_obj.ath_usd and primary_price_obj.ath_usd > 0:
                current = primary_price_obj.price_usd
                ath = primary_price_obj.ath_usd
                pct_from_ath = ((current / ath) - 1) * 100   # negative below ATH
                gain_to_ath = ((ath / current) - 1) * 100    # always positive
                is_ath_q = any(phrase in q_lower for phrase in _ATH_PHRASES)
                ath_context = AthContext(
                    symbol=primary_symbol,
                    ath_usd=ath,
                    current_usd=current,
                    pct_from_ath=pct_from_ath,
                    gain_to_ath_pct=gain_to_ath,
                    is_ath_question=is_ath_q,
                )

    signals = MarketSignals(
        market_id=market.id,
        question=market.question,
        event_title=market.event_title,
        implied_prob=market.yes_price,
        days_to_resolution=days,
        volume_24hr=market.volume_24hr,
        total_volume=market.volume,
        liquidity=market.liquidity,
        spread=market.spread,
        taker_fee=market.taker_fee,
        break_even_edge=market.taker_fee,
        spot_prices=price_map,
        price_changes_24h=change_24h_map,
        price_changes_7d=change_7d_map,
        market_caps=market_cap_map,
        fear_greed_value=fear_greed.value if fear_greed else None,
        fear_greed_classification=fear_greed.classification if fear_greed else None,
        ath_context=ath_context,
        coin_high_24h=coin_high_24h,
        coin_low_24h=coin_low_24h,
        volume_momentum=volume_momentum,
    )

    signals.price_target = _extract_price_target(market.question, price_map)
    return signals


def _extract_price_target(question: str, price_map: dict[str, float]) -> PriceTarget | None:
    """
    Parse a price target from a question like 'Will BTC hit $200k by 2026?'
    Returns None if no price target is found or the relevant coin isn't tracked.
    """
    q_lower = question.lower()

    symbol: str | None = None
    for kw, sym in _COIN_KEYWORDS.items():
        if kw in q_lower:
            symbol = sym
            break

    if symbol is None or symbol not in price_map:
        return None

    current = price_map[symbol]
    if current <= 0:
        return None

    # Match patterns: $200k, $200,000, $1m, $1.5m, $1b
    pattern = r"\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|m|b)?"
    match = re.search(pattern, q_lower)
    if not match:
        return None

    raw = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    target = raw * multipliers.get(suffix, 1)

    if target <= 0:
        return None

    gain_pct = (target / current - 1) * 100
    return PriceTarget(
        symbol=symbol,
        target_usd=target,
        current_usd=current,
        gain_needed_pct=gain_pct,
        multiple_needed=target / current,
    )
