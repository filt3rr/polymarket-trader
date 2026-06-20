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


@dataclass
class PriceTarget:
    """Extracted price target from a market question."""
    symbol: str
    target_usd: float
    current_usd: float
    gain_needed_pct: float       # (target / current - 1) * 100
    multiple_needed: float       # target / current


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

    # Macro sentiment
    fear_greed_value: int | None = None
    fear_greed_classification: str | None = None

    # Price target (populated only when question references a price level)
    price_target: PriceTarget | None = None

    # Derived momentum: 24h vol as % of average daily vol over market lifetime
    volume_momentum: float | None = None


def build_signals(
    market: GammaMarket,
    prices: list[CryptoPrice],
    fear_greed: FearGreedData | None,
) -> MarketSignals:
    price_map = {p.symbol: p.price_usd for p in prices}
    change_map = {p.symbol: p.change_24h for p in prices}

    days = market.days_to_resolution

    # Volume momentum: ratio of 24h vol to average daily vol over market life
    volume_momentum: float | None = None
    if days is not None and market.volume > 0 and days > 0:
        # Approximate age: we don't store created_at, so use volume as proxy
        # If 24h vol > average daily vol → recent surge
        avg_daily = market.volume / max(days, 1)
        if avg_daily > 0:
            volume_momentum = market.volume_24hr / avg_daily

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
        price_changes_24h=change_map,
        fear_greed_value=fear_greed.value if fear_greed else None,
        fear_greed_classification=fear_greed.classification if fear_greed else None,
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

    # Identify the coin referenced
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
