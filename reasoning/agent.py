"""
Probability estimation agent.

Builds a structured prompt from MarketSignals, calls Claude with a tool-use
constraint to get typed output, and returns a ProbabilityEstimate.
"""
import logging
from datetime import datetime, timezone

import anthropic

from reasoning.schemas import ProbabilityEstimate, TradeSignal
from reasoning.signals import MarketSignals

logger = logging.getLogger(__name__)

# ── Tool definition ───────────────────────────────────────────────────────────

_ESTIMATE_TOOL: dict = {
    "name": "submit_estimate",
    "description": (
        "Submit your calibrated probability estimate for the YES outcome "
        "of this prediction market."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "model_probability": {
                "type": "number",
                "description": "P(YES) — your best estimate (0.0 to 1.0).",
            },
            "confidence_low": {
                "type": "number",
                "description": "Lower bound of your 90% confidence interval.",
            },
            "confidence_high": {
                "type": "number",
                "description": "Upper bound of your 90% confidence interval.",
            },
            "reasoning": {
                "type": "string",
                "description": "2-4 sentence chain-of-thought explaining your estimate.",
            },
            "key_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 key risks or uncertainties that could invalidate your estimate.",
            },
        },
        "required": [
            "model_probability",
            "confidence_low",
            "confidence_high",
            "reasoning",
            "key_risks",
        ],
    },
}

# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(signals: MarketSignals) -> str:
    lines: list[str] = []

    lines.append("PREDICTION MARKET PROBABILITY ESTIMATION")
    lines.append("")
    lines.append(f'Question:  "{signals.question}"')
    if signals.event_title and signals.event_title != signals.question:
        lines.append(f"Event:     {signals.event_title}")
    lines.append("")

    lines.append("MARKET SIGNALS")
    lines.append(f"  Implied probability (market consensus):  {signals.implied_prob:.1%}")
    if signals.days_to_resolution is not None:
        lines.append(f"  Days to resolution:                      {signals.days_to_resolution:.0f}")
    lines.append(f"  24h volume:                              ${signals.volume_24hr:,.0f}")
    lines.append(f"  Total volume:                            ${signals.total_volume:,.0f}")
    lines.append(f"  Liquidity:                               ${signals.liquidity:,.0f}")
    if signals.spread is not None:
        lines.append(f"  Spread:                                  {signals.spread:.1%}")
    lines.append(f"  Taker fee (cost to enter):               {signals.taker_fee:.1%}")
    lines.append(f"  Minimum gross edge to profit:            {signals.break_even_edge:.1%}")
    if signals.volume_momentum is not None:
        trend = "above" if signals.volume_momentum > 1 else "below"
        lines.append(
            f"  Volume momentum:                         {signals.volume_momentum:.2f}x "
            f"average daily ({trend} historical pace)"
        )
    lines.append("")

    if signals.spot_prices:
        lines.append("CRYPTO SPOT PRICES (at scan time)")
        for sym, price in sorted(signals.spot_prices.items()):
            chg = signals.price_changes_24h.get(sym, 0.0)
            sign = "+" if chg >= 0 else ""
            lines.append(f"  {sym:<6}  ${price:>13,.2f}  {sign}{chg:.2f}% (24h)")
        lines.append("")

    if signals.fear_greed_value is not None:
        lines.append(
            f"MARKET SENTIMENT  Fear & Greed Index:  "
            f"{signals.fear_greed_value}/100  ({signals.fear_greed_classification})"
        )
        lines.append("")

    if signals.price_target is not None:
        pt = signals.price_target
        lines.append("PRICE TARGET ANALYSIS")
        lines.append(f"  Coin:            {pt.symbol}")
        lines.append(f"  Current price:   ${pt.current_usd:,.2f}")
        lines.append(f"  Target price:    ${pt.target_usd:,.2f}")
        lines.append(f"  Required gain:   {pt.gain_needed_pct:+.1f}%  ({pt.multiple_needed:.2f}x)")
        if signals.days_to_resolution:
            lines.append(
                f"  Time window:     {signals.days_to_resolution:.0f} days "
                f"({signals.days_to_resolution / 365:.2f} years)"
            )
        lines.append("")

    lines.append("TASK")
    lines.append(
        "Using the signals above and your knowledge of crypto market dynamics, "
        "estimate the probability that this market resolves YES."
    )
    lines.append("")
    lines.append(
        "Calibration guidance: the market's implied probability reflects collective "
        "wisdom. Deviate only where the quantitative signals or your knowledge provide "
        "clear evidence. Avoid anchoring — reason independently, then check against "
        "the implied probability."
    )
    lines.append("")
    lines.append("Call submit_estimate with your result.")

    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ProbabilityAgent:
    """
    Calls Claude with a tool-use constraint to obtain a structured
    ProbabilityEstimate for a single market.
    """

    def __init__(self, api_key: str, model: str, min_net_edge: float = 0.10) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._min_net_edge = min_net_edge

    def estimate(self, signals: MarketSignals) -> ProbabilityEstimate:
        """
        Synchronous estimate call. Returns a HOLD estimate on failure after
        3 attempts rather than raising.
        """
        prompt = _build_prompt(signals)

        for attempt in range(1, 4):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    tools=[_ESTIMATE_TOOL],
                    tool_choice={"type": "any"},
                    messages=[{"role": "user", "content": prompt}],
                )
                return self._parse_response(response, signals)
            except Exception as exc:
                logger.warning(
                    "Estimate attempt %d/3 failed for market %s: %s",
                    attempt,
                    signals.market_id,
                    exc,
                )

        logger.error("All attempts failed for market %s — returning HOLD.", signals.market_id)
        return self._fallback_hold(signals)

    def _parse_response(
        self, response: anthropic.types.Message, signals: MarketSignals
    ) -> ProbabilityEstimate:
        tool_use = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use is None:
            raise ValueError("No tool_use block in response")

        args = tool_use.input
        model_prob = float(args["model_probability"])
        implied = signals.implied_prob
        gross_edge = model_prob - implied
        net_edge = gross_edge - signals.taker_fee

        # Determine signal direction
        if net_edge >= self._min_net_edge:
            signal = TradeSignal.BUY_YES
        elif -(net_edge + 2 * signals.taker_fee) >= self._min_net_edge:
            # NO is underpriced: model_prob << implied_prob
            signal = TradeSignal.BUY_NO
        else:
            signal = TradeSignal.HOLD

        return ProbabilityEstimate(
            market_id=signals.market_id,
            question=signals.question,
            model_probability=model_prob,
            implied_probability=implied,
            gross_edge=gross_edge,
            taker_fee=signals.taker_fee,
            net_edge=net_edge,
            confidence_low=float(args["confidence_low"]),
            confidence_high=float(args["confidence_high"]),
            signal=signal,
            reasoning=args["reasoning"],
            key_risks=args.get("key_risks", []),
            llm_model=self._model,
            estimated_at=datetime.now(tz=timezone.utc),
        )

    def _fallback_hold(self, signals: MarketSignals) -> ProbabilityEstimate:
        implied = signals.implied_prob
        return ProbabilityEstimate(
            market_id=signals.market_id,
            question=signals.question,
            model_probability=implied,
            implied_probability=implied,
            gross_edge=0.0,
            taker_fee=signals.taker_fee,
            net_edge=-signals.taker_fee,
            confidence_low=max(0.0, implied - 0.15),
            confidence_high=min(1.0, implied + 0.15),
            signal=TradeSignal.HOLD,
            reasoning="Estimation failed after 3 attempts — defaulting to HOLD.",
            key_risks=["LLM call failed"],
            llm_model=self._model,
            estimated_at=datetime.now(tz=timezone.utc),
        )
