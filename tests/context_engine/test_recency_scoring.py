"""Tests for adaptive recency scoring in RAG."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from context_engine.rag_manager import recency_score


class TestRecencyScore:
    def test_recent_entity_high_score(self):
        """Entity mentioned 1 hour ago should have high recency score."""
        last_mentioned = datetime.now(timezone.utc) - timedelta(hours=1)
        score = recency_score(last_mentioned, mention_count=1, confidence=0.8)
        assert score > 0.9

    def test_old_entity_low_score(self):
        """Entity mentioned 7 days ago should have low recency score."""
        last_mentioned = datetime.now(timezone.utc) - timedelta(days=7)
        score = recency_score(last_mentioned, mention_count=1, confidence=0.5)
        assert score < 0.5

    def test_3_day_half_life(self):
        """At 3 days, score should be meaningfully decayed but not gone.
        Effective half-life with mention_count=1, confidence=0.5 is ~108 hours,
        so at 72 hours the score is ~0.63."""
        last_mentioned = datetime.now(timezone.utc) - timedelta(days=3)
        score = recency_score(last_mentioned, mention_count=1, confidence=0.5)
        assert 0.4 < score < 0.75

    def test_more_mentions_slower_decay(self):
        """Frequently mentioned entities should decay slower."""
        last_mentioned = datetime.now(timezone.utc) - timedelta(days=3)
        score_1 = recency_score(last_mentioned, mention_count=1, confidence=0.5)
        score_5 = recency_score(last_mentioned, mention_count=5, confidence=0.5)
        assert score_5 > score_1

    def test_higher_confidence_slower_decay(self):
        """Higher confidence entities should decay slower."""
        last_mentioned = datetime.now(timezone.utc) - timedelta(days=3)
        score_low = recency_score(last_mentioned, mention_count=1, confidence=0.3)
        score_high = recency_score(last_mentioned, mention_count=1, confidence=0.9)
        assert score_high > score_low

    def test_future_timestamp_clamped(self):
        """Future timestamps should still return valid score."""
        last_mentioned = datetime.now(timezone.utc) + timedelta(hours=1)
        score = recency_score(last_mentioned, mention_count=1, confidence=0.5)
        assert 0.0 <= score <= 1.0

    def test_score_always_between_0_and_1(self):
        """Score should always be in [0, 1] range."""
        for days in [0, 0.1, 1, 3, 7, 14, 30, 90]:
            for mentions in [1, 3, 5, 10]:
                for conf in [0.3, 0.5, 0.7, 0.95]:
                    ts = datetime.now(timezone.utc) - timedelta(days=days)
                    score = recency_score(ts, mention_count=mentions, confidence=conf)
                    assert 0.0 <= score <= 1.0, f"days={days} mentions={mentions} conf={conf} score={score}"
