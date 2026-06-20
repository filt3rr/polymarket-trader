"""
Paper position management.

Determines fill price for YES/NO entries and provides helpers used by
the scanner and resolution checker.
"""
from data.models import GammaMarket
from reasoning.schemas import ProbabilityEstimate, TradeSignal


def fill_price_for_estimate(estimate: ProbabilityEstimate, market: GammaMarket) -> float:
    """
    Simulate entry price:
      BUY_YES → pay the YES ask (best price to buy YES shares)
      BUY_NO  → pay the NO ask  = 1 - YES bid
    Falls back to yes_price / (1 - yes_price) when live book is absent.
    """
    if estimate.signal == TradeSignal.BUY_YES:
        price = market.best_ask if market.best_ask is not None else market.yes_price
    else:
        bid = market.best_bid if market.best_bid is not None else market.yes_price
        price = 1.0 - bid

    # Clamp to valid probability range (avoid division-by-zero on degenerate markets)
    return max(0.01, min(0.99, price))


def direction_for_estimate(estimate: ProbabilityEstimate) -> str:
    return "YES" if estimate.signal == TradeSignal.BUY_YES else "NO"
