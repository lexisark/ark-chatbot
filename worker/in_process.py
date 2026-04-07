"""In-process job queue using asyncio tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class InProcessQueue:
    """Simple job queue that runs jobs as asyncio tasks. No external deps."""

    async def enqueue(self, job_type: str, job_fn: Callable[[], Awaitable]) -> None:
        asyncio.create_task(self._run(job_type, job_fn))

    async def _run(self, job_type: str, job_fn: Callable[[], Awaitable]) -> None:
        try:
            await job_fn()
            logger.debug(f"Job {job_type} completed")
        except Exception:
            logger.exception(f"Job {job_type} failed")
