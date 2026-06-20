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
        coin_ids = ",".join(TRACKED_COINS.keys())
        resp = await self._client.get(
            "/simple/price",
            params={
                "ids": coin_ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        now = datetime.now(tz=timezone.utc)

        prices: list[CryptoPrice] = []
        for coin_id, symbol in TRACKED_COINS.items():
            if coin_id not in data:
                continue
            d = data[coin_id]
            prices.append(
                CryptoPrice(
                    symbol=symbol,
                    price_usd=float(d.get("usd", 0)),
                    change_24h=float(d.get("usd_24h_change", 0)),
                    volume_24h_usd=float(d.get("usd_24h_vol", 0)),
                    scanned_at=now,
                )
            )

        logger.info("CoinGecko: fetched prices for %d coins", len(prices))
        return prices
