import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import Settings
from data.gamma import GammaClient
from data.coingecko import CoinGeckoClient
from data.sentiment import SentimentClient
from data.models import CryptoPrice, FearGreedData, GammaMarket, ScanResult
from db.models import Base
from db import queries
from reasoning.signals import build_signals
from reasoning.schemas import ProbabilityEstimate, TradeSignal
from reasoning.agent import ProbabilityAgent
from trading.executor import TradeExecutor

logger = logging.getLogger(__name__)

_COL_WIDTH = 76


class MarketScanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_async_engine(settings.database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self._agent = ProbabilityAgent(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            min_net_edge=settings.min_net_edge,
        )
        self._executor = TradeExecutor(settings)

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await queries.migrate_db(self._engine)
        logger.info("Database tables created / verified.")

    async def scan(self) -> ScanResult:
        scanned_at = datetime.now(tz=timezone.utc)

        # Merge env settings with any runtime config overrides
        async with self._session_factory() as _cfg_session:
            overrides = await queries.get_runtime_config(_cfg_session)
        effective_min_edge = float(overrides.get("min_net_edge", self._settings.min_net_edge))
        effective_max_markets = int(overrides.get("max_markets_per_scan", self._settings.max_markets_per_scan))
        effective_position_size = float(overrides.get("paper_position_size_usd", self._settings.paper_position_size_usd))
        effective_min_volume = float(overrides.get("min_market_volume", self._settings.min_market_volume))
        self._agent.min_net_edge = effective_min_edge

        logger.info(
            "Scan starting at %s  [mode=%s edge=%.2f max_mkts=%d stake=$%.0f vol_min=$%.0f]",
            scanned_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            self._executor.mode_label,
            effective_min_edge, effective_max_markets,
            effective_position_size, effective_min_volume,
        )

        # ── 1. Fetch market data in parallel ─────────────────────────────────
        async with (
            GammaClient(
                base_url=self._settings.gamma_base_url,
                page_size=self._settings.gamma_market_limit,
            ) as gamma,
            CoinGeckoClient(api_key=self._settings.coingecko_api_key) as cg,
            SentimentClient() as sentiment,
        ):
            markets_res, prices_res, fg_res = await asyncio.gather(
                gamma.get_crypto_markets(
                    keywords=self._settings.keyword_list,
                    min_volume=effective_min_volume,
                    min_liquidity=self._settings.min_market_liquidity,
                ),
                cg.get_prices(),
                sentiment.get_fear_greed(),
                return_exceptions=True,
            )

        markets: list[GammaMarket] = markets_res if not isinstance(markets_res, Exception) else []
        prices: list[CryptoPrice] = prices_res if not isinstance(prices_res, Exception) else []
        fg: FearGreedData | None = fg_res if not isinstance(fg_res, Exception) else None

        if isinstance(markets_res, Exception):
            logger.error("Gamma API error: %s", markets_res)
        if isinstance(prices_res, Exception):
            logger.error("CoinGecko error: %s", prices_res)
        if isinstance(fg_res, Exception):
            logger.error("Fear & Greed error: %s", fg_res)

        # ── 2. Persist raw market data ────────────────────────────────────────
        async with self._session_factory() as session:
            async with session.begin():
                for market in markets:
                    await queries.upsert_market(session, market)
                    await queries.save_market_snapshot(session, market, scanned_at)
                if prices:
                    await queries.save_price_snapshots(session, prices)
                if fg:
                    await queries.save_sentiment_snapshot(session, fg, scanned_at)

        # ── 3. Run probability model on top N markets ─────────────────────────
        estimates: list[ProbabilityEstimate] = []
        estimate_rows = []
        if markets and self._settings.anthropic_api_key:
            candidates = sorted(markets, key=lambda m: m.volume, reverse=True)
            candidates = candidates[:effective_max_markets]
            logger.info("Running probability model on %d markets...", len(candidates))

            for market in candidates:
                signals = build_signals(market, prices, fg)
                est = await asyncio.to_thread(self._agent.estimate, signals)
                estimates.append(est)
                logger.debug("  %s", est)

            async with self._session_factory() as session:
                async with session.begin():
                    for est in estimates:
                        row = await queries.save_probability_estimate(session, est)
                        estimate_rows.append((est, row))

        # ── 4. Execute positions for new opportunities ────────────────────────
        if estimate_rows:
            async with self._session_factory() as session:
                existing_market_ids = await queries.get_open_position_market_ids(session)

            new_pos_count = 0
            for est, est_row in estimate_rows:
                if not est.is_opportunity:
                    continue
                if est.market_id in existing_market_ids:
                    continue
                market = next((m for m in markets if m.id == est.market_id), None)
                if market is None:
                    continue

                exec_result = await self._executor.execute(est, market, effective_position_size)
                if not exec_result.success:
                    logger.error(
                        "Execution failed for '%s': %s",
                        market.question[:50], exec_result.error,
                    )
                    continue

                async with self._session_factory() as session:
                    async with session.begin():
                        await queries.open_position(
                            session,
                            market=market,
                            estimate_row=est_row,
                            direction=exec_result.direction,
                            fill_price=exec_result.fill_price,
                            num_shares=exec_result.num_shares,
                            stake_usdc=exec_result.stake_usdc,
                            is_live=exec_result.is_live,
                            order_id=exec_result.order_id,
                            order_status=exec_result.order_status,
                        )
                new_pos_count += 1
                existing_market_ids.add(est.market_id)
                logger.info(
                    "[%s] %s %s @ $%.3f",
                    self._executor.mode_label,
                    exec_result.direction,
                    market.question[:50],
                    exec_result.fill_price,
                )

            if new_pos_count:
                logger.info("Opened %d new position(s) [%s].", new_pos_count, self._executor.mode_label)

        result = ScanResult(
            scanned_at=scanned_at,
            markets=markets,
            prices=prices,
            fear_greed=fg,
        )
        self._print_summary(result, estimates)
        return result

    # ── Console output ────────────────────────────────────────────────────────

    def _print_summary(
        self, result: ScanResult, estimates: list[ProbabilityEstimate]
    ) -> None:
        ts = result.scanned_at.strftime("%Y-%m-%d %H:%M UTC")
        mode = self._executor.mode_label
        print(f"\n{'=' * _COL_WIDTH}")
        print(f"  POLYMARKET SCAN  {ts}  [{mode}]")
        print(f"{'=' * _COL_WIDTH}")
        self._print_prices(result.prices)
        self._print_fear_greed(result.fear_greed)
        self._print_markets(result.markets)
        self._print_estimates(estimates)
        print()

    def _print_prices(self, prices: list[CryptoPrice]) -> None:
        if not prices:
            return
        print("\n  SPOT PRICES")
        for p in sorted(prices, key=lambda x: x.price_usd, reverse=True):
            sign = "+" if p.change_24h >= 0 else ""
            print(
                f"    {p.symbol:<6}  ${p.price_usd:>13,.2f}  "
                f"{sign}{p.change_24h:+.2f}% (24h)"
            )

    def _print_fear_greed(self, fg: FearGreedData | None) -> None:
        if fg is None:
            return
        bar_filled = round(fg.value / 5)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        print(f"\n  FEAR & GREED  [{bar}]  {fg.value}/100  {fg.classification}")

    def _print_markets(self, markets: list[GammaMarket]) -> None:
        if not markets:
            print("\n  No crypto markets found matching criteria.")
            return
        top = sorted(markets, key=lambda m: m.volume, reverse=True)[:25]
        print(f"\n  CRYPTO MARKETS  ({len(markets)} total, top {len(top)} by volume)\n")
        header = (
            f"  {'YES':>5}  {'NO':>5}  {'BID':>5}  {'ASK':>5}  "
            f"{'24H VOL':>9}  {'LIQ':>8}  {'DAYS':>4}  QUESTION"
        )
        print(header)
        print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*4}  {'-'*38}")
        for m in top:
            days = f"{m.days_to_resolution:.0f}" if m.days_to_resolution is not None else "  --"
            bid = f"{m.best_bid:.2f}" if m.best_bid is not None else "   --"
            ask = f"{m.best_ask:.2f}" if m.best_ask is not None else "   --"
            q = m.question[:58] + "..." if len(m.question) > 58 else m.question
            print(
                f"  {m.yes_price:>5.2f}  {m.no_price:>5.2f}  "
                f"{bid:>5}  {ask:>5}  "
                f"${m.volume_24hr:>8,.0f}  "
                f"${m.liquidity:>7,.0f}  "
                f"{days:>4}  {q}"
            )

    def _print_estimates(self, estimates: list[ProbabilityEstimate]) -> None:
        if not estimates:
            return
        opportunities = [e for e in estimates if e.is_opportunity]
        print(f"\n  PROBABILITY MODEL  ({len(estimates)} markets analysed)\n")
        header = (
            f"  {'SIGNAL':<9}  {'MODEL':>6}  {'MARKET':>6}  "
            f"{'EDGE':>8}  {'CI 90%':^13}  QUESTION"
        )
        print(header)
        print(f"  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*13}  {'-'*38}")
        for e in sorted(estimates, key=lambda x: abs(x.trade_edge), reverse=True):
            flag = " *" if e.is_opportunity else "  "
            ci = f"{e.confidence_low:.0%}-{e.confidence_high:.0%}"
            q = e.question[:48] + "..." if len(e.question) > 48 else e.question
            print(
                f"  {e.signal.value:<9}  {e.model_probability:>6.1%}  "
                f"{e.implied_probability:>6.1%}  {e.trade_edge:>+8.1%}  "
                f"{ci:^13}  {q}{flag}"
            )
        if opportunities:
            print(f"\n  OPPORTUNITIES  ({len(opportunities)} flagged)\n")
            for e in opportunities:
                direction = "YES" if e.signal == TradeSignal.BUY_YES else "NO"
                print(f"  [{direction}]  edge {e.trade_edge:+.1%}  --  {e.question}")
                print(f"         {e.reasoning[:120]}")
                if e.key_risks:
                    risks = " | ".join(e.key_risks[:2])
                    print(f"         Risks: {risks}")
                print()
