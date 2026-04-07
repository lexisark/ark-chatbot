"""Tests for in-process job queue."""

import asyncio

import pytest

from worker.base import JobQueue
from worker.in_process import InProcessQueue


class TestInProcessQueue:
    def test_implements_protocol(self):
        q = InProcessQueue()
        assert isinstance(q, JobQueue)

    async def test_enqueue_runs_job(self):
        q = InProcessQueue()
        results = []

        async def job():
            results.append("done")

        await q.enqueue("test_job", job)
        # Give the task time to run
        await asyncio.sleep(0.1)

        assert results == ["done"]

    async def test_job_failure_doesnt_crash(self):
        q = InProcessQueue()

        async def bad_job():
            raise ValueError("boom")

        # Should not raise
        await q.enqueue("bad_job", bad_job)
        await asyncio.sleep(0.1)

    async def test_multiple_jobs(self):
        q = InProcessQueue()
        results = []

        async def job(value):
            results.append(value)

        await q.enqueue("job1", lambda: job("a"))
        await q.enqueue("job2", lambda: job("b"))
        await asyncio.sleep(0.2)

        assert set(results) == {"a", "b"}
