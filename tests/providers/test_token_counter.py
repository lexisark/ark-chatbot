"""Tests for token counter implementations."""

from providers.token_counter import CharacterEstimateCounter, TiktokenCounter


class TestTiktokenCounter:
    def setup_method(self):
        self.counter = TiktokenCounter()

    def test_count_returns_positive_int(self):
        result = self.counter.count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_empty_string(self):
        assert self.counter.count("") == 0

    def test_count_messages_includes_overhead(self):
        messages = [{"role": "user", "content": "hi"}]
        token_count = self.counter.count_messages(messages)
        content_only = self.counter.count("hi")
        # Should be more than just the content due to role + overhead
        assert token_count > content_only

    def test_count_messages_multiple(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = self.counter.count_messages(messages)
        assert result > 0

    def test_truncate_long_text(self):
        long_text = "This is a long sentence that should be truncated to fit within a small token budget."
        truncated = self.counter.truncate(long_text, 5)
        truncated_tokens = self.counter.count(truncated)
        assert truncated_tokens <= 5

    def test_truncate_short_text_unchanged(self):
        short = "hi"
        result = self.counter.truncate(short, 100)
        assert result == short

    def test_fits_within_budget(self):
        assert self.counter.fits("short", 100) is True

    def test_does_not_fit_budget(self):
        long_text = "word " * 200
        assert self.counter.fits(long_text, 1) is False

    def test_count_is_deterministic(self):
        text = "the quick brown fox"
        assert self.counter.count(text) == self.counter.count(text)


class TestCharacterEstimateCounter:
    def setup_method(self):
        self.counter = CharacterEstimateCounter()

    def test_count_roughly_4_chars_per_token(self):
        text = "abcdefgh"  # 8 chars → ~2 tokens
        result = self.counter.count(text)
        assert result == 2

    def test_count_empty(self):
        # Min 1 for non-empty, 0 for empty
        assert self.counter.count("") == 0

    def test_count_messages(self):
        messages = [{"role": "user", "content": "hello"}]
        result = self.counter.count_messages(messages)
        assert result > 0

    def test_truncate(self):
        text = "a" * 100
        truncated = self.counter.truncate(text, 5)
        # 5 tokens * 4 chars = 20 chars max
        assert len(truncated) <= 20

    def test_truncate_short_unchanged(self):
        assert self.counter.truncate("hi", 100) == "hi"

    def test_fits(self):
        assert self.counter.fits("hi", 100) is True
        assert self.counter.fits("a" * 1000, 1) is False
