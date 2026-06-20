"""
Phase 6 — Risk Management Engine.

Centralises all pre-trade risk checks and Kelly-based position sizing.
Called by MarketScanner before every execution attempt.
"""
import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from config import Settings
from data.models import GammaMarket
from reasoning.schemas import ProbabilityEstimate, TradeSignal

logger = logging.getLogger(__name__)

# Maps canonical token name → search keywords (question text, lower-cased)
_CRYPTO_TOKENS: dict[str, list[str]] = {
    "btc":   ["btc", "bitcoin"],
    "eth":   ["eth", "ethereum"],
    "sol":   ["sol", "solana"],
    "bnb":   ["bnb", "binance"],
    "xrp":   ["xrp", "ripple"],
    "ada":   ["ada", "cardano"],
    "avax":  ["avax", "avalanche"],
    "doge":  ["doge", "dogecoin"],
    "link":  ["chainlink"],
    "dot":   ["polkadot"],
    "matic": ["matic", "polygon"],
    "ltc":   ["ltc", "litecoin"],
    "pepe":  ["pepe"],
    "shib":  ["shiba"],
    "sui":   ["sui"],
}


@dataclass
class RiskCheck:
    """Result of a single named risk gate."""
    name: str
    passed: bool
    detail: str


@dataclass
class RiskResult:
    approved: bool
    stake_usdc: float
    reject_reason: str = ""
    kelly_f_full: float = 0.0       # full (un-fractioned) Kelly fraction
    kelly_f_used: float = 0.0       # fractional Kelly actually applied
    checks: list[RiskCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "stake_usdc": self.stake_usdc,
            "reject_reason": self.reject_reason,
            "kelly_f_full": round(self.kelly_f_full, 4),
            "kelly_f_used": round(self.kelly_f_used, 4),
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


class RiskEngine:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    # ── Public API ────────────────────────────────────────────────────────────

    async def check(
        self,
        session: AsyncSession,
        estimate: ProbabilityEstimate,
        market: GammaMarket,
        base_stake: float,
    ) -> RiskResult:
        """
        Run all risk gates and return a RiskResult with the approved stake.
        Checks run in order — first failure short-circuits the rest.
        """
        from db import queries
        params = await self._effective_params(session)
        checks: list[RiskCheck] = []

        # ── Gate 1: Circuit breaker ───────────────────────────────────────────
        if params["drawdown_limit_usd"] > 0:
            stats = await queries.get_calibration_stats(session)
            realized_pnl = stats["net_paper_pnl"] or 0.0
            unrealized_pnl = await queries.get_total_unrealized_pnl(session)
            total_pnl = realized_pnl + unrealized_pnl
            limit = params["drawdown_limit_usd"]
            if total_pnl <= -limit:
                c = RiskCheck("circuit_breaker", False,
                              f"TRIPPED — total P&L ${total_pnl:+.2f} ≤ -${limit:.2f}")
                checks.append(c)
                logger.warning("Risk: circuit breaker tripped (P&L $%.2f)", total_pnl)
                return RiskResult(approved=False, stake_usdc=0.0,
                                  reject_reason=c.detail, checks=checks)
            checks.append(RiskCheck("circuit_breaker", True,
                                    f"P&L ${total_pnl:+.2f}  threshold -${limit:.2f}"))
        else:
            checks.append(RiskCheck("circuit_breaker", True, "disabled"))

        # ── Gate 2: Max concurrent positions ─────────────────────────────────
        open_count = await queries.get_open_position_count(session)
        max_pos = params["max_open_positions"]
        if open_count >= max_pos:
            c = RiskCheck("max_positions", False,
                          f"BLOCKED — {open_count}/{max_pos} open")
            checks.append(c)
            logger.info("Risk: max positions reached (%d/%d)", open_count, max_pos)
            return RiskResult(approved=False, stake_usdc=0.0,
                              reject_reason=c.detail, checks=checks)
        checks.append(RiskCheck("max_positions", True, f"{open_count}/{max_pos}"))

        # ── Gate 3: Max total exposure ────────────────────────────────────────
        max_exp = params["max_exposure_usd"]
        if max_exp > 0:
            exposure = await queries.get_total_open_exposure(session)
            if exposure + base_stake > max_exp:
                c = RiskCheck("max_exposure", False,
                              f"BLOCKED — ${exposure:.0f} deployed + ${base_stake:.0f} stake > ${max_exp:.0f} limit")
                checks.append(c)
                logger.info("Risk: exposure limit (current $%.0f / max $%.0f)", exposure, max_exp)
                return RiskResult(approved=False, stake_usdc=0.0,
                                  reject_reason=c.detail, checks=checks)
            checks.append(RiskCheck("max_exposure", True,
                                    f"${exposure:.0f} / ${max_exp:.0f}"))
        else:
            checks.append(RiskCheck("max_exposure", True, "disabled"))

        # ── Gate 4: Per-underlying concentration ─────────────────────────────
        max_conc = params["max_positions_per_underlying"]
        underlying = _detect_underlying(market.question)
        if max_conc > 0 and underlying:
            token, keywords = underlying
            conc = await queries.get_open_positions_for_underlying(session, keywords)
            if conc >= max_conc:
                c = RiskCheck("concentration", False,
                              f"BLOCKED — {conc}/{max_conc} {token.upper()} positions open")
                checks.append(c)
                logger.info("Risk: concentration limit for %s (%d/%d)", token, conc, max_conc)
                return RiskResult(approved=False, stake_usdc=0.0,
                                  reject_reason=c.detail, checks=checks)
            checks.append(RiskCheck("concentration", True,
                                    f"{token.upper()}: {conc}/{max_conc}"))
        else:
            checks.append(RiskCheck("concentration", True,
                                    "no underlying detected" if not underlying else "disabled"))

        # ── Gate 5: Bid/ask spread ────────────────────────────────────────────
        max_spread = params["max_spread_pct"]
        if max_spread > 0 and market.best_bid is not None and market.best_ask is not None:
            spread = market.best_ask - market.best_bid
            mid = (market.best_ask + market.best_bid) / 2.0
            spread_pct = spread / mid if mid > 0 else 1.0
            if spread_pct > max_spread:
                c = RiskCheck("spread", False,
                              f"BLOCKED — spread {spread_pct:.1%} > {max_spread:.1%} limit")
                checks.append(c)
                logger.info("Risk: spread too wide %.1f%% > %.1f%%", spread_pct * 100, max_spread * 100)
                return RiskResult(approved=False, stake_usdc=0.0,
                                  reject_reason=c.detail, checks=checks)
            checks.append(RiskCheck("spread", True, f"{spread_pct:.1%} ≤ {max_spread:.1%}"))
        else:
            checks.append(RiskCheck("spread", True,
                                    "skipped (no bid/ask data)" if (market.best_bid is None) else "disabled"))

        # ── Kelly sizing ──────────────────────────────────────────────────────
        kelly_f_full = _kelly_fraction(estimate)
        kf = params["kelly_fraction"]          # 0 means fixed sizing
        max_size = params["max_position_size_usd"]
        bankroll = params["paper_bankroll_usd"]

        if kf > 0 and kelly_f_full > 0:
            kelly_f_used = kelly_f_full * kf
            stake = kelly_f_used * bankroll
        else:
            kelly_f_used = 0.0
            stake = base_stake

        stake = min(stake, max_size) if max_size > 0 else stake
        stake = max(stake, 1.0)
        stake = round(stake, 2)

        checks.append(RiskCheck(
            "kelly_sizing", True,
            f"f*={kelly_f_full:.3f}  ×{kf}  bankroll=${bankroll:.0f}  → ${stake:.2f}"
            if kf > 0 else f"fixed sizing → ${stake:.2f}",
        ))

        logger.info(
            "Risk: APPROVED %s  stake=$%.2f  kelly=%.3f (%.0f%% fraction)  [%d gates passed]",
            market.question[:40], stake, kelly_f_full, kf * 100, len(checks),
        )

        return RiskResult(
            approved=True,
            stake_usdc=stake,
            kelly_f_full=kelly_f_full,
            kelly_f_used=kelly_f_full * kf if kf > 0 else 0.0,
            checks=checks,
        )

    async def get_status(self, session: AsyncSession) -> dict:
        """Snapshot dict for the dashboard risk card."""
        from db import queries
        params = await self._effective_params(session)
        stats = await queries.get_calibration_stats(session)
        realized_pnl = stats["net_paper_pnl"] or 0.0
        unrealized_pnl = await queries.get_total_unrealized_pnl(session)
        total_pnl = realized_pnl + unrealized_pnl
        open_count = await queries.get_open_position_count(session)
        exposure = await queries.get_total_open_exposure(session)
        limit = params["drawdown_limit_usd"]
        circuit_tripped = limit > 0 and total_pnl <= -limit
        return {
            "circuit_tripped": circuit_tripped,
            "circuit_threshold_usd": limit,
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "open_positions": open_count,
            "max_open_positions": params["max_open_positions"],
            "total_exposure_usd": exposure,
            "max_exposure_usd": params["max_exposure_usd"],
            "kelly_fraction": params["kelly_fraction"],
            "paper_bankroll_usd": params["paper_bankroll_usd"],
            "max_position_size_usd": params["max_position_size_usd"],
            "max_spread_pct": params["max_spread_pct"],
            "max_positions_per_underlying": params["max_positions_per_underlying"],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _effective_params(self, session: AsyncSession) -> dict:
        """Merge env defaults with runtime config DB overrides."""
        from db import queries
        overrides = await queries.get_runtime_config(session)
        s = self._s

        def _f(key: str, default: float) -> float:
            return float(overrides[key]) if key in overrides else default

        def _i(key: str, default: int) -> int:
            return int(float(overrides[key])) if key in overrides else default

        def _b(key: str, default: bool) -> bool:
            if key in overrides:
                return overrides[key].lower() in ("true", "1", "yes")
            return default

        return {
            "drawdown_limit_usd":          _f("drawdown_limit_usd",          s.drawdown_limit_usd),
            "max_open_positions":          _i("max_open_positions",          s.max_open_positions),
            "max_exposure_usd":            _f("max_exposure_usd",            s.max_exposure_usd),
            "max_position_size_usd":       _f("max_position_size_usd",       s.max_position_size_usd),
            "kelly_fraction":              _f("kelly_fraction",              s.kelly_fraction),
            "paper_bankroll_usd":          _f("paper_bankroll_usd",          s.paper_bankroll_usd),
            "max_positions_per_underlying": _i("max_positions_per_underlying", s.max_positions_per_underlying),
            "max_spread_pct":              _f("max_spread_pct",              s.max_spread_pct),
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _kelly_fraction(estimate: ProbabilityEstimate) -> float:
    """
    Full Kelly fraction (0–1 of bankroll) for a prediction market position.

    BUY_YES: f* = (model_prob - mkt_prob) / (1 - mkt_prob)
    BUY_NO:  f* = (mkt_prob - model_prob) / mkt_prob
    """
    p = estimate.model_probability
    mkt = estimate.implied_probability

    if estimate.signal == TradeSignal.BUY_YES:
        denom = 1.0 - mkt
        f = (p - mkt) / denom if denom > 0 else 0.0
    else:
        f = (mkt - p) / mkt if mkt > 0 else 0.0

    return max(0.0, min(f, 1.0))


def _detect_underlying(question: str) -> tuple[str, list[str]] | None:
    """Return (token_name, keywords) for the dominant crypto in the question, or None."""
    q = question.lower()
    for token, keywords in _CRYPTO_TOKENS.items():
        if any(kw in q for kw in keywords):
            return token, keywords
    return None
