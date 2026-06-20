"""
Telegram alert integration — no library dependency, pure httpx.

Setup:
  1. Create a bot via @BotFather → copy the token
  2. Start a chat with your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Copy the chat_id from the response
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""
import logging

import httpx

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    _API.format(token=self._token),
                    json={
                        "chat_id": self._chat_id,
                        "text": message,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                )
                if resp.status_code != 200:
                    logger.warning("Telegram send failed (%s): %s", resp.status_code, resp.text[:200])
                    return False
            return True
        except Exception as exc:
            logger.warning("Telegram error: %s", exc)
            return False

    # ── Alert templates ───────────────────────────────────────────────────────

    async def alert_new_position(
        self,
        direction: str,
        question: str,
        stake: float,
        edge: float,
        kelly_f: float,
        mode: str,
    ) -> None:
        emoji = "🟢" if direction == "YES" else "🟡"
        tag = "📝 PAPER" if mode == "PAPER" else "💸 LIVE"
        kelly_str = f"{kelly_f:.1%} Kelly" if kelly_f > 0 else "fixed size"
        await self.send(
            f"{tag} {emoji} <b>New Position</b>\n\n"
            f"<b>BUY {direction}</b> | ${stake:.2f} USDC | edge {edge:+.1%} | {kelly_str}\n\n"
            f"<i>{question[:280]}</i>"
        )

    async def alert_position_resolved(
        self, direction: str, question: str, pnl: float, outcome_str: str
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        result = "WIN" if pnl >= 0 else "LOSS"
        await self.send(
            f"{emoji} <b>Position Resolved — {result}</b>\n\n"
            f"P&L: <b>{'+'if pnl>=0 else ''}${pnl:.2f}</b> | {direction} | outcome: {outcome_str}\n\n"
            f"<i>{question[:280]}</i>"
        )

    async def alert_circuit_breaker_tripped(self, total_pnl: float, threshold: float) -> None:
        await self.send(
            f"⛔ <b>Circuit Breaker TRIPPED</b>\n\n"
            f"Total P&amp;L <b>${total_pnl:+.2f}</b> hit the -${threshold:.0f} threshold.\n"
            f"New positions are <b>paused</b>. Adjust threshold in Config or wait for P&amp;L recovery."
        )

    async def alert_circuit_breaker_reset(self, total_pnl: float) -> None:
        await self.send(
            f"✅ <b>Circuit Breaker Reset</b>\n\n"
            f"Total P&amp;L recovered to <b>${total_pnl:+.2f}</b>. Trading has resumed."
        )

    async def alert_component_down(self, display_name: str, detail: str) -> None:
        await self.send(
            f"🚨 <b>{display_name} — DOWN</b>\n\n"
            f"{detail}\n\n"
            f"<i>New positions are paused if this is a required component.</i>"
        )

    async def alert_component_recovered(self, display_name: str) -> None:
        await self.send(f"✅ <b>{display_name} — Recovered</b>")

    async def alert_low_balance(self, usdc: float, threshold: float) -> None:
        await self.send(
            f"⚠️ <b>Low Wallet Balance</b>\n\n"
            f"USDC: <b>${usdc:.2f}</b> (threshold: ${threshold:.2f})\n"
            f"Consider topping up before the next live trade."
        )

    async def alert_daily_summary(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        open_count: int,
        win_count: int,
        loss_count: int,
    ) -> None:
        total = realized_pnl + unrealized_pnl
        emoji = "📈" if total >= 0 else "📉"
        wr = f"{win_count/(win_count+loss_count)*100:.0f}%" if (win_count + loss_count) else "N/A"
        await self.send(
            f"{emoji} <b>Daily Summary</b>\n\n"
            f"Realized: {'+'if realized_pnl>=0 else ''}${realized_pnl:.2f}\n"
            f"Unrealized: {'+'if unrealized_pnl>=0 else ''}${unrealized_pnl:.2f}\n"
            f"<b>Total: {'+'if total>=0 else ''}${total:.2f}</b>\n\n"
            f"Open positions: {open_count} | W/L: {win_count}/{loss_count} ({wr})"
        )
