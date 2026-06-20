"""
Trade execution router.

Routes each opportunity to paper simulation or live CLOB order
based on PAPER_TRADING flag and wallet availability.
Keeps the scanner and position-opening logic clean of mode-switching.
"""
import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    direction: str           # YES | NO
    fill_price: float
    num_shares: float
    stake_usdc: float
    is_live: bool
    order_id: str | None = None
    order_status: str = "SIMULATED"   # SIMULATED | PENDING | FILLED | CANCELLED
    error: str | None = None


class TradeExecutor:
    """
    Instantiated once by MarketScanner at startup.
    In paper mode: simulates fills immediately with no external calls.
    In live mode:  places GTC limit orders via Polymarket CLOB.
    Falls back to paper mode silently if live setup fails.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._clob = None
        self._wallet = None

        if not settings.paper_trading and settings.polygon_private_key:
            try:
                from trading.clob import ClobExecutor
                from trading.wallet import PolygonWallet
                self._clob = ClobExecutor(settings.polygon_private_key)
                self._wallet = PolygonWallet(
                    settings.polygon_private_key,
                    rpc_url=settings.polygon_rpc_url,
                )
                logger.info(
                    "Live trading ENABLED  wallet=%s",
                    self._wallet.address_masked,
                )
            except Exception as exc:
                logger.error(
                    "Live trading init failed — falling back to PAPER: %s", exc
                )
                self._clob = None
                self._wallet = None

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """True only if CLOB is connected and PAPER_TRADING is False."""
        return self._clob is not None and not self._settings.paper_trading

    @property
    def mode_label(self) -> str:
        return "LIVE" if self.is_live else "PAPER"

    @property
    def wallet_address(self) -> str | None:
        return self._wallet.address if self._wallet else None

    @property
    def wallet_address_masked(self) -> str | None:
        return self._wallet.address_masked if self._wallet else None

    # ── Wallet status (async, used by dashboard) ──────────────────────────────

    async def get_wallet_status(self) -> dict | None:
        """Returns wallet balance dict or None in paper mode / on error."""
        if self._wallet is None or self._clob is None:
            return None
        try:
            usdc, pol, clob_ok = await asyncio.gather(
                asyncio.to_thread(self._wallet.usdc_balance),
                asyncio.to_thread(self._wallet.pol_balance),
                asyncio.to_thread(self._clob.health_check),
            )
            return {
                "address": self._wallet.address,
                "address_masked": self._wallet.address_masked,
                "usdc_balance": round(usdc, 2),
                "pol_balance": round(pol, 4),
                "clob_connected": clob_ok,
            }
        except Exception as exc:
            logger.warning("Wallet status check failed: %s", exc)
            return None

    # ── Core execution ────────────────────────────────────────────────────────

    async def execute(
        self,
        estimate,       # ProbabilityEstimate schema object
        market,         # GammaMarket dataclass
        stake_usdc: float,
    ) -> ExecutionResult:
        from paper.positions import direction_for_estimate, fill_price_for_estimate
        from reasoning.schemas import TradeSignal

        direction = direction_for_estimate(estimate)
        fill_price = fill_price_for_estimate(estimate, market)
        num_shares = stake_usdc / fill_price

        # ── Paper simulation ──────────────────────────────────────────────────
        if not self.is_live:
            logger.info(
                "[PAPER] %s %s @ $%.4f  shares=%.2f  stake=$%.2f",
                direction, market.question[:45], fill_price, num_shares, stake_usdc,
            )
            return ExecutionResult(
                success=True,
                direction=direction,
                fill_price=fill_price,
                num_shares=num_shares,
                stake_usdc=stake_usdc,
                is_live=False,
                order_id=None,
                order_status="SIMULATED",
            )

        # ── Live CLOB order ───────────────────────────────────────────────────
        token_id = (
            market.yes_token_id if estimate.signal == TradeSignal.BUY_YES
            else market.no_token_id
        )
        if not token_id:
            msg = f"No token ID for {direction} side of market {market.id}"
            logger.error("[LIVE] %s", msg)
            return ExecutionResult(
                success=False,
                direction=direction,
                fill_price=fill_price,
                num_shares=num_shares,
                stake_usdc=stake_usdc,
                is_live=True,
                error=msg,
            )

        try:
            result = await asyncio.to_thread(
                self._clob.place_limit_order,
                token_id,
                fill_price,
                num_shares,
                "BUY",
            )
            order_status = "FILLED" if result.is_filled else "PENDING"
            logger.info(
                "[LIVE] %s %s @ $%.4f  shares=%.2f  order=%.12s  status=%s",
                direction, market.question[:45], fill_price, num_shares,
                result.order_id, order_status,
            )
            return ExecutionResult(
                success=True,
                direction=direction,
                fill_price=fill_price,
                num_shares=num_shares,
                stake_usdc=stake_usdc,
                is_live=True,
                order_id=result.order_id,
                order_status=order_status,
            )
        except Exception as exc:
            logger.error("[LIVE] Order failed for %s: %s", market.id, exc)
            return ExecutionResult(
                success=False,
                direction=direction,
                fill_price=fill_price,
                num_shares=num_shares,
                stake_usdc=stake_usdc,
                is_live=True,
                error=str(exc),
            )
