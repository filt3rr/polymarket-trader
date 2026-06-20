"""Unit tests for data model parsing — no network calls required."""
import json
import pytest
from datetime import datetime, timezone
from data.models import GammaMarket, CryptoPrice, FearGreedData


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_raw_market(**overrides) -> dict:
    base = {
        "id": "12345",
        "question": "Will BTC hit $200k in 2026?",
        "description": "Bitcoin price prediction market.",
        "conditionId": "0xabc",
        "outcomePrices": '["0.42", "0.58"]',
        "clobTokenIds": '["111111111", "222222222"]',
        "volumeNum": 250_000.0,
        "volume24hr": 12_000.0,
        "liquidityNum": 15_000.0,
        "endDate": "2026-12-31T23:59:59Z",
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "negRisk": False,
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5.0,
        "bestBid": 0.41,
        "bestAsk": 0.43,
        "spread": 0.02,
        "lastTradePrice": 0.42,
        "feeSchedule": {"rate": 0.05, "takerOnly": True, "rebateRate": 0.25},
        "events": [{"title": "Bitcoin 2026 Predictions"}],
    }
    base.update(overrides)
    return base


# ── GammaMarket.from_api ──────────────────────────────────────────────────────

class TestGammaMarketParsing:
    def test_basic_parse(self):
        m = GammaMarket.from_api(_make_raw_market())
        assert m.id == "12345"
        assert m.question == "Will BTC hit $200k in 2026?"
        assert m.yes_price == pytest.approx(0.42)
        assert m.no_price == pytest.approx(0.58)
        assert m.yes_token_id == "111111111"
        assert m.no_token_id == "222222222"
        assert m.volume == 250_000.0
        assert m.volume_24hr == 12_000.0
        assert m.liquidity == 15_000.0
        assert m.accepting_orders is True
        assert m.neg_risk is False
        assert m.taker_fee == pytest.approx(0.05)
        assert m.best_bid == pytest.approx(0.41)
        assert m.best_ask == pytest.approx(0.43)
        assert m.spread == pytest.approx(0.02)
        assert m.event_title == "Bitcoin 2026 Predictions"

    def test_end_date_parsed_as_utc(self):
        m = GammaMarket.from_api(_make_raw_market())
        assert m.end_date is not None
        assert m.end_date.tzinfo is not None
        assert m.end_date.year == 2026

    def test_missing_end_date(self):
        m = GammaMarket.from_api(_make_raw_market(endDate=None))
        assert m.end_date is None
        assert m.days_to_resolution is None

    def test_outcomePrices_as_list(self):
        # Already a list instead of JSON string
        m = GammaMarket.from_api(_make_raw_market(outcomePrices=["0.70", "0.30"]))
        assert m.yes_price == pytest.approx(0.70)
        assert m.no_price == pytest.approx(0.30)

    def test_clobTokenIds_as_list(self):
        m = GammaMarket.from_api(_make_raw_market(clobTokenIds=["abc", "def"]))
        assert m.yes_token_id == "abc"
        assert m.no_token_id == "def"

    def test_missing_events(self):
        m = GammaMarket.from_api(_make_raw_market(events=None))
        assert m.event_title == ""

    def test_mid_price_from_bid_ask(self):
        m = GammaMarket.from_api(_make_raw_market())
        assert m.mid_price == pytest.approx(0.42)

    def test_mid_price_fallback_to_yes_price(self):
        m = GammaMarket.from_api(_make_raw_market(bestBid=None, bestAsk=None))
        assert m.mid_price == m.yes_price

    def test_days_to_resolution_positive(self):
        m = GammaMarket.from_api(_make_raw_market())
        # end date is in future
        assert m.days_to_resolution is not None
        assert m.days_to_resolution > 0


# ── Keyword filter logic ──────────────────────────────────────────────────────

class TestKeywordFilter:
    """Simulates the filter logic in GammaClient.get_crypto_markets."""

    KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"]

    def _matches(self, question: str, description: str = "") -> bool:
        text = (question + " " + description).lower()
        return any(kw in text for kw in self.KEYWORDS)

    def test_btc_question_matches(self):
        assert self._matches("Will BTC hit $200k?")

    def test_bitcoin_matches(self):
        assert self._matches("Bitcoin price above $150k by end of year?")

    def test_eth_matches(self):
        assert self._matches("Will ETH flip Bitcoin in 2026?")

    def test_solana_matches(self):
        assert self._matches("Will Solana reach $1000?")

    def test_case_insensitive(self):
        assert self._matches("BITCOIN to $300K")

    def test_non_crypto_does_not_match(self):
        assert not self._matches("Will Rihanna release a new album?")

    def test_crypto_in_description_matches(self):
        assert self._matches("Will X happen?", description="Crypto market prediction.")


# ── CryptoPrice ───────────────────────────────────────────────────────────────

class TestCryptoPrice:
    def test_construction(self):
        p = CryptoPrice(
            symbol="BTC",
            price_usd=105_000.0,
            change_24h=-2.5,
            volume_24h_usd=35_000_000_000.0,
            scanned_at=datetime.now(tz=timezone.utc),
        )
        assert p.symbol == "BTC"
        assert p.price_usd == 105_000.0
        assert p.change_24h == -2.5


# ── FearGreedData ─────────────────────────────────────────────────────────────

class TestFearGreedData:
    def test_construction(self):
        fg = FearGreedData(
            value=72,
            classification="Greed",
            timestamp=datetime.now(tz=timezone.utc),
        )
        assert 0 <= fg.value <= 100
        assert fg.classification == "Greed"
