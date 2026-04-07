"""Data structures for the context engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YouRememberBlock:
    """Formatted memory block for the LLM system instruction."""

    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    recaps: list[str] = field(default_factory=list)
    total_tokens: int = 0
    episode_budget_ratio: float = 0.0


@dataclass
class ContextAssemblyResult:
    """Result of context assembly process."""

    system_prompt: str = ""
    system_prompt_tokens: int = 0
    memories_text: str = ""
    memories_tokens: int = 0
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    recent_tokens: int = 0
    total_tokens: int = 0
    assembly_time_ms: int = 0
    system_instruction: str = ""
