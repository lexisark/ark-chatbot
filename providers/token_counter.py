"""Token counter implementations."""

import tiktoken

from .registry import registry


class TiktokenCounter:
    """Token counter using tiktoken (works for OpenAI models, approximate for others)."""

    def __init__(self, encoding: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message structure overhead
            total += self.count(msg.get("content", ""))
            total += 1  # role token
        total += 2  # conversation overhead
        return total

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens])

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget


class CharacterEstimateCounter:
    """Rough token counter (~4 chars per token). No dependencies."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return sum(self.count(m.get("content", "")) + 5 for m in messages)

    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget


registry.register_counter("tiktoken", TiktokenCounter)
registry.register_counter("character_estimate", CharacterEstimateCounter)
