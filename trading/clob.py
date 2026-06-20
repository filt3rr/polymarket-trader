"""
Polymarket CLOB API client — order placement, fill monitoring, cancellation.
Wraps py-clob-client with structured results and clean error handling.
"""
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# Import side constants with fallback for version differences
try:
    from py_clob_client.order_builder.constants import BUY, SELL  # type: ignore
except ImportError:
    BUY = "BUY"  # type: ignore
    SELL = "SELL"  # type: ignore


@dataclass
class OrderResult:
    order_id: str
    status: str           # "live" | "matched" | "delayed" | "cancelled"
    price: float
    size: float
    token_id: str
    side: Literal["BUY", "SELL"]

    @property
    def is_filled(self) -> bool:
        return self.status in ("matched", "MATCHED", "filled", "FILLED")

    @property
    def is_live(self) -> bool:
        return self.status in ("live", "LIVE", "delayed", "DELAYED")


class ClobExecutor:
    """
    Thin wrapper around py-clob-client.
    All blocking calls must be wrapped in asyncio.to_thread() by the caller.
    """

    def __init__(self, private_key: str) -> None:
        from py_clob_client.client import ClobClient  # type: ignore

        self._client = ClobClient(
            host=CLOB_HOST,
            key=private_key,
            chain_id=POLYGON_CHAIN_ID,
            signature_type=0,   # 0 = EOA (Externally Owned Account)
        )
        creds = self._client.create_or_derive_api_creds()
        self._client.set_api_creds(creds)
        logger.info("CLOB client initialised (host=%s)", CLOB_HOST)

    # ── Order placement ───────────────────────────────────────────────────────

    def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: Literal["BUY", "SELL"] = "BUY",
    ) -> OrderResult:
        """
        Place a GTC limit order on the CLOB.
        price: cents on the dollar (0.0–1.0)
        size:  number of shares
        Returns OrderResult with order_id and fill status.
        """
        from py_clob_client.clob_types import OrderArgs  # type: ignore

        side_const = BUY if side == "BUY" else SELL
        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(size, 2),
            side=side_const,
        )
        resp = self._client.create_and_post_order(order_args)
        order_id = (
            resp.get("orderID")
            or resp.get("order_id")
            or resp.get("id")
            or ""
        )
        status = resp.get("status", "live")
        logger.info(
            "Order placed  side=%s token=%.12s price=%.4f size=%.2f  "
            "order_id=%.12s status=%s",
            side, token_id, price, size, order_id, status,
        )
        return OrderResult(
            order_id=order_id,
            status=status,
            price=price,
            size=size,
            token_id=token_id,
            side=side,
        )

    # ── Order queries ─────────────────────────────────────────────────────────

    def get_order_status(self, order_id: str) -> dict:
        """Return raw order dict from CLOB API."""
        return self._client.get_order(order_id) or {}

    def get_fills(self, order_id: str) -> list[dict]:
        """Return list of fill/trade dicts for an order."""
        try:
            from py_clob_client.clob_types import TradeParams  # type: ignore
            params = TradeParams(id=order_id)
            resp = self._client.get_trades(params)
            if isinstance(resp, dict):
                return resp.get("data", [])
            if isinstance(resp, list):
                return resp
            return []
        except Exception as exc:
            logger.warning("get_fills(%s): %s", order_id[:12], exc)
            return []

    def get_average_fill_price(self, order_id: str) -> float | None:
        """Compute volume-weighted average fill price from trade history."""
        fills = self.get_fills(order_id)
        if not fills:
            return None
        total_value = sum(float(f.get("price", 0)) * float(f.get("size", 0)) for f in fills)
        total_size = sum(float(f.get("size", 0)) for f in fills)
        return total_value / total_size if total_size > 0 else None

    # ── Order management ──────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel(order_id)
            logger.info("Order cancelled: %s", order_id[:12])
            return True
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id[:12], exc)
            return False

    def health_check(self) -> bool:
        try:
            result = self._client.get_ok()
            return bool(result)
        except Exception:
            return False
