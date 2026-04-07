"""Provider registry — maps names to factory classes, resolved at runtime."""

from __future__ import annotations

from typing import Any


class ProviderRegistry:
    def __init__(self):
        self._chat_factories: dict[str, type] = {}
        self._embedding_factories: dict[str, type] = {}
        self._counter_factories: dict[str, type] = {}

    # ── Registration ────────────────────────────────────

    def register_chat(self, name: str, factory: type) -> None:
        self._chat_factories[name] = factory

    def register_embedding(self, name: str, factory: type) -> None:
        self._embedding_factories[name] = factory

    def register_counter(self, name: str, factory: type) -> None:
        self._counter_factories[name] = factory

    # ── Creation ────────────────────────────────────────

    def create_chat(self, name: str, **kwargs: Any):
        if name not in self._chat_factories:
            raise KeyError(f"Unknown chat provider: '{name}'. Available: {list(self._chat_factories.keys())}")
        return self._chat_factories[name](**kwargs)

    def create_embedding(self, name: str, **kwargs: Any):
        if name not in self._embedding_factories:
            raise KeyError(f"Unknown embedding provider: '{name}'. Available: {list(self._embedding_factories.keys())}")
        return self._embedding_factories[name](**kwargs)

    def create_counter(self, name: str, **kwargs: Any):
        if name not in self._counter_factories:
            raise KeyError(f"Unknown token counter: '{name}'. Available: {list(self._counter_factories.keys())}")
        return self._counter_factories[name](**kwargs)

    # ── Listing ─────────────────────────────────────────

    def list_chat_providers(self) -> list[str]:
        return list(self._chat_factories.keys())

    def list_embedding_providers(self) -> list[str]:
        return list(self._embedding_factories.keys())

    def list_counter_providers(self) -> list[str]:
        return list(self._counter_factories.keys())


# Global registry
registry = ProviderRegistry()
