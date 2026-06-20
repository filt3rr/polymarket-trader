import logging
from datetime import datetime, timezone
import httpx
from .models import CryptoPrice

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Coins to track — maps CoinGecko ID → display symbol
TRACKED_COINS: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "ripple": "XRP",
    "binancecoin": "BNB",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "chainlink": "LINK",
    "avalanche-2": "AVAX",
    "polygon-ecosystem-token": "POL",
}


class CoinGeckoClient:
    """CoinGecko free-tier (demo) price client."""

    def __init__(self, api_key: str = "") -> None:
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["x-cg-demo-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=COINGECKO_BASE,
            headers=headers,
            timeout=30,
        )

    async def __aenter__(self) -> "CoinGeckoClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self._client.aclose()

    async def get_prices(self) -> list[CryptoPrice]:
        """
        Fetch enriched price data via /coins/markets.

        Returns per coin: spot price, 24h/7d change, market cap,
        24h high/low, and all-time high — all in a single API call.
        Previously used /simple/price which only provided spot + 24h change.
        """
        coin_ids = ",".join(TRACKED_COINS.keys())
        resp = await self._client.get(
            "/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": coin_ids,
                "order": "market_cap_desc",
                "per_page": len(TRACKED_COINS),
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "7d",
            },
        )
        resp.raise_for_status()
        coins = resp.json()
        now = datetime.now(tz=timezone.utc)

        prices: list[CryptoPrice] = []
        for coin in coins:
            coin_id = coin.get("id", "")
            symbol = TRACKED_COINS.get(coin_id)
            if symbol is None:
                continue

            ath = coin.get("ath")
            ath_chg = coin.get("ath_change_percentage")

            prices.append(
                CryptoPrice(
                    symbol=symbol,
                    price_usd=float(coin.get("current_price") or 0),
                    change_24h=float(coin.get("price_change_percentage_24h") or 0),
                    change_7d=_safe_float(coin.get("price_change_percentage_7d_in_currency")),
                    volume_24h_usd=float(coin.get("total_volume") or 0),
                    market_cap_usd=_safe_float(coin.get("market_cap")),
                    high_24h=_safe_float(coin.get("high_24h")),
                    low_24h=_safe_float(coin.get("low_24h")),
                    ath_usd=_safe_float(ath),
                    ath_change_pct=_safe_float(ath_chg),
                    scanned_at=now,
                )
            )

        logger.info("CoinGecko /coins/markets: fetched enriched data for %d coins", len(prices))
        return prices


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
