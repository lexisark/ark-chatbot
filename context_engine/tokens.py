"""Token counting and budget management utilities."""

from __future__ import annotations

from providers.base import TokenCounter


class TokenHelper:
    """Wraps a TokenCounter protocol with context-engine-specific helpers."""

    def __init__(self, counter: TokenCounter):
        self._counter = counter

    def count(self, text: str) -> int:
        return self._counter.count(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return self._counter.count_messages(messages)

    def truncate(self, text: str, max_tokens: int) -> str:
        return self._counter.truncate(text, max_tokens)

    def fits(self, text: str, budget: int) -> bool:
        return self._counter.fits(text, budget)

    def fit_messages_to_budget(
        self, messages: list[dict[str, str]], budget: int
    ) -> list[dict[str, str]]:
        """Keep the most recent messages that fit within the token budget.

        Walks backwards from the most recent message, adding messages
        until the budget is exceeded. Returns in chronological order.
        """
        if not messages:
            return []

        selected: list[dict[str, str]] = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self._counter.count_messages([msg])
            if current_tokens + msg_tokens > budget:
                break
            selected.append(msg)
            current_tokens += msg_tokens

        selected.reverse()
        return selected
