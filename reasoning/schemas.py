from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator


class TradeSignal(str, Enum):
    BUY_YES = "BUY_YES"   # model thinks YES is underpriced
    BUY_NO = "BUY_NO"     # model thinks NO is underpriced
    HOLD = "HOLD"         # edge below threshold or high uncertainty


class ProbabilityEstimate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    market_id: str
    question: str

    # Probabilities
    model_probability: float      # P(YES) as estimated by the model
    implied_probability: float    # P(YES) as priced by the market

    # Edge (YES-perspective)
    gross_edge: float             # model_probability - implied_probability
    taker_fee: float              # cost to enter as taker (e.g. 0.05)
    net_edge: float               # gross_edge - taker_fee

    # Uncertainty bounds (90% confidence interval)
    confidence_low: float
    confidence_high: float

    # Decision
    signal: TradeSignal
    reasoning: str
    key_risks: list[str]

    # Metadata
    llm_model: str
    estimated_at: datetime

    @field_validator("model_probability", "implied_probability",
                     "confidence_low", "confidence_high")
    @classmethod
    def clamp_probability(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @property
    def is_opportunity(self) -> bool:
        return self.signal != TradeSignal.HOLD

    @property
    def confidence_width(self) -> float:
        return self.confidence_high - self.confidence_low

    @property
    def trade_edge(self) -> float:
        """Edge in the direction of the signal — always positive for opportunities."""
        if self.signal == TradeSignal.BUY_NO:
            return -(self.net_edge + 2 * self.taker_fee)
        return self.net_edge

    def __str__(self) -> str:
        return (
            f"[{self.signal.value:<8}]  "
            f"model={self.model_probability:.1%}  "
            f"market={self.implied_probability:.1%}  "
            f"trade_edge={self.trade_edge:+.1%}  "
            f"{self.question[:60]}"
        )
