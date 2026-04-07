"""Job queue protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class JobQueue(Protocol):
    async def enqueue(self, job_type: str, job_fn: Callable[[], Awaitable]) -> None: ...
