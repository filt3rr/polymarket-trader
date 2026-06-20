"""Unit tests for signal computation — no network or LLM calls."""
import pytest
from datetime import datetime, timezone
from data.models import CryptoPrice, FearGreedData, GammaMarket
from reasoning.signals import build_signals, _extract_price_target, MarketSignals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_market(**overrides) -> GammaMarket:
    import json
    base = {
        "id": "99999",
        "question": "Will BTC hit $200k by end of 2026?",
        "description": "Bitcoin price target market.",
        "conditionId": "0xabc",
        "outcomePrices": '["0.42", "0.58"]',
        "clobTokenIds": '["111", "222"]',
        "volumeNum": 500_000.0,
        "volume24hr": 25_000.0,
        "liquidityNum": 20_000.0,
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
        "events": [{"title": "BTC 2026"}],
    }
    base.update(overrides)
    return GammaMarket.from_api(base)


def _prices() -> list[CryptoPrice]:
    now = datetime.now(tz=timezone.utc)
    return [
        CryptoPrice(symbol="BTC", price_usd=63_000.0, change_24h=1.5,
                    volume_24h_usd=30_000_000_000.0, scanned_at=now),
        CryptoPrice(symbol="ETH", price_usd=1_700.0, change_24h=0.8,
                    volume_24h_usd=10_000_000_000.0, scanned_at=now),
    ]


def _fg() -> FearGreedData:
    return FearGreedData(
        value=23,
        classification="Extreme Fear",
        timestamp=datetime.now(tz=timezone.utc),
    )


# ── build_signals ─────────────────────────────────────────────────────────────

class TestBuildSignals:
    def test_returns_market_signals(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert isinstance(s, MarketSignals)

    def test_implied_prob_matches_market(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.implied_prob == pytest.approx(0.42)

    def test_spot_prices_populated(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.spot_prices["BTC"] == pytest.approx(63_000.0)
        assert s.spot_prices["ETH"] == pytest.approx(1_700.0)

    def test_fear_greed_populated(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.fear_greed_value == 23
        assert s.fear_greed_classification == "Extreme Fear"

    def test_taker_fee_from_market(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.taker_fee == pytest.approx(0.05)
        assert s.break_even_edge == pytest.approx(0.05)

    def test_volume_momentum_computed(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.volume_momentum is not None
        assert s.volume_momentum > 0

    def test_no_fear_greed(self):
        s = build_signals(_make_market(), _prices(), None)
        assert s.fear_greed_value is None
        assert s.fear_greed_classification is None

    def test_empty_prices(self):
        s = build_signals(_make_market(), [], _fg())
        assert s.spot_prices == {}
        assert s.price_target is None  # no prices → can't compute target


# ── price target extraction ───────────────────────────────────────────────────

class TestExtractPriceTarget:
    PRICES = {"BTC": 63_000.0, "ETH": 1_700.0}

    def test_btc_dollar_k_suffix(self):
        pt = _extract_price_target("Will BTC hit $200k by 2026?", self.PRICES)
        assert pt is not None
        assert pt.symbol == "BTC"
        assert pt.target_usd == pytest.approx(200_000.0)
        assert pt.current_usd == pytest.approx(63_000.0)
        assert pt.multiple_needed == pytest.approx(200_000 / 63_000)

    def test_btc_full_number(self):
        pt = _extract_price_target("Will Bitcoin reach $150,000?", self.PRICES)
        assert pt is not None
        assert pt.target_usd == pytest.approx(150_000.0)

    def test_eth_target(self):
        pt = _extract_price_target("Will ETH hit $10k?", self.PRICES)
        assert pt is not None
        assert pt.symbol == "ETH"
        assert pt.target_usd == pytest.approx(10_000.0)

    def test_million_suffix(self):
        pt = _extract_price_target("Will Bitcoin hit $1m?", self.PRICES)
        assert pt is not None
        assert pt.target_usd == pytest.approx(1_000_000.0)

    def test_no_price_in_question(self):
        pt = _extract_price_target("Will BTC ETH flip?", self.PRICES)
        assert pt is None

    def test_unknown_coin(self):
        pt = _extract_price_target("Will SHIB hit $0.001?", self.PRICES)
        assert pt is None  # SHIB not in price_map

    def test_gain_needed_above_current(self):
        pt = _extract_price_target("Will BTC hit $200k?", self.PRICES)
        assert pt is not None
        assert pt.gain_needed_pct > 0  # target above current price

    def test_gain_needed_below_current(self):
        # Target below current price — a "crash" market
        pt = _extract_price_target("Will BTC drop to $30k?", self.PRICES)
        assert pt is not None
        assert pt.gain_needed_pct < 0


# ── MarketSignals.price_target integration ────────────────────────────────────

class TestPriceTargetInSignals:
    def test_price_target_populated_for_price_market(self):
        s = build_signals(_make_market(), _prices(), _fg())
        assert s.price_target is not None
        assert s.price_target.symbol == "BTC"
        assert s.price_target.target_usd == pytest.approx(200_000.0)

    def test_no_price_target_for_non_price_market(self):
        m = _make_market(question="Will Ethereum merge proof-of-stake?")
        s = build_signals(m, _prices(), _fg())
        assert s.price_target is None
