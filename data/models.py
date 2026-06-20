import json
from datetime import datetime
from pydantic import BaseModel, field_validator


class GammaMarket(BaseModel):
    id: str
    question: str
    description: str | None = None
    # Probability prices (0.0–1.0)
    yes_price: float
    no_price: float
    # Live CLOB prices
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    last_trade_price: float | None = None
    # Market metadata
    volume: float
    volume_24hr: float
    liquidity: float
    end_date: datetime | None = None
    active: bool
    accepting_orders: bool
    neg_risk: bool = False
    min_tick: float = 0.01
    min_order_size: float = 5.0
    taker_fee: float = 0.05
    # Token IDs needed for CLOB order placement
    yes_token_id: str
    no_token_id: str
    condition_id: str = ""
    # Parent event label (for display grouping)
    event_title: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "GammaMarket":
        prices_raw = data.get("outcomePrices", '["0.5","0.5"]')
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw

        tokens_raw = data.get("clobTokenIds", '["",""]')
        tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw

        events = data.get("events") or []
        event_title = events[0].get("title", "") if events else ""

        fee_schedule = data.get("feeSchedule") or {}
        taker_fee = float(fee_schedule.get("rate", 0.05))

        end_raw = data.get("endDate")
        end_date: datetime | None = None
        if end_raw:
            # Strip trailing Z and parse as UTC-aware
            end_date = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))

        return cls(
            id=str(data["id"]),
            question=data.get("question", ""),
            description=data.get("description"),
            yes_price=float(prices[0]) if prices else 0.5,
            no_price=float(prices[1]) if len(prices) > 1 else 0.5,
            best_bid=data.get("bestBid"),
            best_ask=data.get("bestAsk"),
            spread=data.get("spread"),
            last_trade_price=data.get("lastTradePrice"),
            volume=float(data.get("volumeNum", 0)),
            volume_24hr=float(data.get("volume24hr", 0)),
            liquidity=float(data.get("liquidityNum", 0)),
            end_date=end_date,
            active=bool(data.get("active", False)),
            accepting_orders=bool(data.get("acceptingOrders", False)),
            neg_risk=bool(data.get("negRisk", False)),
            min_tick=float(data.get("orderPriceMinTickSize", 0.01)),
            min_order_size=float(data.get("orderMinSize", 5.0)),
            taker_fee=taker_fee,
            yes_token_id=tokens[0] if tokens else "",
            no_token_id=tokens[1] if len(tokens) > 1 else "",
            condition_id=data.get("conditionId", ""),
            event_title=event_title,
        )

    @property
    def days_to_resolution(self) -> float | None:
        if self.end_date is None:
            return None
        from datetime import timezone
        delta = self.end_date - datetime.now(tz=timezone.utc)
        return max(0.0, delta.total_seconds() / 86_400)

    @property
    def mid_price(self) -> float:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.yes_price


class CryptoPrice(BaseModel):
    symbol: str
    price_usd: float
    change_24h: float
    volume_24h_usd: float
    scanned_at: datetime


class FearGreedData(BaseModel):
    value: int
    classification: str
    timestamp: datetime


class ScanResult(BaseModel):
    scanned_at: datetime
    markets: list[GammaMarket]
    prices: list[CryptoPrice]
    fear_greed: FearGreedData | None
