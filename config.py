from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Safety — never remove this guard
    paper_trading: bool = True

    # Data sources
    coingecko_api_key: str = ""
    anthropic_api_key: str = ""

    # Gamma API
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    gamma_market_limit: int = 500

    # Scanner
    scan_interval_minutes: int = 15
    min_market_volume: float = 10_000.0
    min_market_liquidity: float = 1_000.0

    # Keyword filter — CSV, matched against question text only
    crypto_keywords: str = (
        "bitcoin,btc,ethereum,eth,crypto,solana,sol,xrp,ripple,"
        "bnb,doge,dogecoin,cardano,ada,chainlink,link,polygon,matic,"
        "avalanche,avax,pepe,shiba,sui,ton,near,aptos"
    )

    # Probability model
    llm_model: str = "claude-sonnet-4-6"
    min_net_edge: float = 0.10
    max_markets_per_scan: int = 10

    # Paper trading
    paper_position_size_usd: float = 10.0

    # Resolution checker
    resolution_check_interval_minutes: int = 30

    # ── Phase 5 — Live execution ──────────────────────────────────────────────
    # Set PAPER_TRADING=False AND provide POLYGON_PRIVATE_KEY to go live.
    # The key is a hex private key for a Polygon EOA wallet (0x...).
    # NEVER commit this value to git — use .env or Railway environment variables.
    polygon_private_key: str = ""

    # Polygon JSON-RPC endpoint. Public RPC works for dev; use Alchemy/Infura in prod.
    polygon_rpc_url: str = "https://polygon-rpc.com"

    # Polymarket CLOB API host
    clob_host: str = "https://clob.polymarket.com"

    # Minutes before a pending live order is cancelled if unfilled
    live_order_timeout_minutes: int = 10

    # Fill monitor job cadence (minutes)
    fill_monitor_interval_minutes: int = 5

    # ── Web UI ───────────────────────────────────────────────────────────────
    web_username: str = "admin"
    web_password: str = "changeme"

    # Database — use /data/ path for Railway persistent volume
    database_url: str = "sqlite+aiosqlite:///polymarket_trader.db"

    # Logging
    log_level: str = "INFO"

    @property
    def keyword_list(self) -> list[str]:
        return [kw.strip().lower() for kw in self.crypto_keywords.split(",") if kw.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
