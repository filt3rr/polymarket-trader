import logging
from datetime import datetime, timezone
import httpx
from .models import FearGreedData

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class SentimentClient:
    """Alternative.me Fear & Greed Index client — no authentication required."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=15)

    async def __aenter__(self) -> "SentimentClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self._client.aclose()

    async def get_fear_greed(self) -> FearGreedData:
        resp = await self._client.get(FEAR_GREED_URL)
        resp.raise_for_status()
        entry = resp.json()["data"][0]
        return FearGreedData(
            value=int(entry["value"]),
            classification=entry["value_classification"],
            timestamp=datetime.fromtimestamp(int(entry["timestamp"]), tz=timezone.utc),
        )
