import logging
import httpx
from .models import GammaMarket

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Keywords that must appear in a question for it to count as a crypto market.
# Used only when purging non-crypto markets that sneak in via shared crypto-tagged events
# (e.g., "Will Jesus Christ return before GTA VI?" ends up in crypto events because
# its sibling market in the same event references Bitcoin).
_CRYPTO_ANCHOR_WORDS = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
    "xrp", "ripple", "bnb", "doge", "dogecoin", "cardano", "ada",
    "chainlink", "link", "polygon", "matic", "avalanche", "avax",
    "pepe", "shiba", "sui", "ton", "near", "aptos", "defi", "nft",
    "web3", "blockchain", "altcoin", "stablecoin", "usdt", "usdc",
    "satoshi", "coinbase", "binance", "kraken", "pump", "airdrop",
})

# GTA VI acts as a timeline anchor across many unrelated markets; any market
# mentioning it that has no crypto anchor word is not a crypto market.
_TIMELINE_ANCHOR_PHRASES = ("gta vi", "gta6", "gta 6")


def _is_non_crypto_contamination(question: str) -> bool:
    """Return True for non-crypto markets that leaked in via crypto-tagged events."""
    q = question.lower()
    if not any(phrase in q for phrase in _TIMELINE_ANCHOR_PHRASES):
        return False
    return not any(kw in q for kw in _CRYPTO_ANCHOR_WORDS)


class GammaClient:
    """Read-only Gamma API client — no authentication required."""

    def __init__(self, base_url: str = GAMMA_BASE, page_size: int = 500) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self) -> "GammaClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self._client.aclose()

    async def _get_all_crypto_events(self) -> list[dict]:
        """
        Paginate through all active events tagged 'crypto' on Polymarket.
        The crypto tag (id=21, slug='crypto') lives at the event level — individual
        markets inherit their category from their parent event.

        The API caps pagination at ~2000 offsets and returns 422 beyond that;
        we stop on any non-2xx response or empty page.
        """
        all_events: list[dict] = []
        offset = 0
        while True:
            try:
                resp = await self._client.get(
                    "/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "tag_slug": "crypto",
                        "limit": 100,
                        "offset": offset,
                    },
                )
                if resp.status_code >= 400:
                    logger.debug("Events pagination stopped at offset=%d (%s)", offset, resp.status_code)
                    break
                page = resp.json()
            except Exception as exc:
                logger.debug("Events pagination stopped at offset=%d: %s", offset, exc)
                break

            if not page:
                break
            all_events.extend(page)
            logger.debug("Events page offset=%d count=%d", offset, len(page))
            if len(page) < 100:
                break
            offset += 100
        logger.info("Total crypto events fetched: %d", len(all_events))
        return all_events

    async def _fetch_markets_page(self, offset: int) -> list[dict]:
        resp = await self._client.get(
            "/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 500,
                "offset": offset,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _get_all_active_markets(self) -> list[dict]:
        """Fallback: paginate through all active markets (no category filter)."""
        all_markets: list[dict] = []
        offset = 0
        while True:
            page = await self._fetch_markets_page(offset)
            if not page:
                break
            all_markets.extend(page)
            if len(page) < 500:
                break
            offset += 500
        logger.info("Total active markets (fallback): %d", len(all_markets))
        return all_markets

    async def get_market_by_id(self, market_id: str) -> dict | None:
        """Fetch a single market — used by the resolution checker."""
        try:
            resp = await self._client.get(f"/markets/{market_id}", timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return data[0] if isinstance(data, list) else data
        except Exception as exc:
            logger.warning("get_market_by_id(%s) failed: %s", market_id, exc)
            return None

    async def get_crypto_markets(
        self,
        keywords: list[str],
        min_volume: float = 10_000.0,
        min_liquidity: float = 1_000.0,
    ) -> list[GammaMarket]:
        """
        Return binary, order-book-enabled crypto markets above the volume threshold.

        Primary strategy: fetch all events tagged 'crypto' via the Gamma /events
        endpoint (tag_slug=crypto), then flatten their nested market lists. This
        surfaces all ~300+ markets that Polymarket itself categorises as Crypto —
        price-range markets, up/down markets, ETF markets, airdrop markets, etc.

        Fallback: if the events endpoint fails, fall back to keyword matching
        against market question text (original approach, ~8 markets).

        Liquidity is not checked separately — event-nested markets often omit
        liquidityNum. Volume is the primary quality gate.
        """
        raw: list[dict] = []
        use_keyword_filter = False

        try:
            events = await self._get_all_crypto_events()
            for event in events:
                for m in (event.get("markets") or []):
                    if isinstance(m, dict):
                        raw.append(m)
            logger.info("Flattened %d markets from %d crypto events", len(raw), len(events))
        except Exception as exc:
            logger.warning("Events approach failed (%s) — falling back to keyword filter", exc)
            raw = await self._get_all_active_markets()
            use_keyword_filter = True

        markets: list[GammaMarket] = []
        seen_ids: set[str] = set()

        for data in raw:
            if not isinstance(data, dict):
                continue
            market_id = str(data.get("id", ""))
            if market_id in seen_ids:
                continue
            seen_ids.add(market_id)

            if data.get("negRisk", False):
                continue
            if not data.get("acceptingOrders", False):
                continue

            if use_keyword_filter:
                question = (data.get("question") or "").lower()
                if not any(kw in question for kw in keywords):
                    continue

            try:
                market = GammaMarket.from_api(data)
            except Exception as exc:
                logger.debug("Skipping market %s: %s", data.get("id"), exc)
                continue

            if market.volume < min_volume:
                continue

            if _is_non_crypto_contamination(market.question):
                logger.debug("Excluded non-crypto contamination: %s", market.question[:60])
                continue

            markets.append(market)

        logger.info(
            "Crypto markets after filtering: %d (source: %s)",
            len(markets),
            "keyword fallback" if use_keyword_filter else "events tag_slug=crypto",
        )
        return markets
