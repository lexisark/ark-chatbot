"""Tests for context engine token utilities."""

from providers.token_counter import TiktokenCounter
from context_engine.tokens import TokenHelper


class TestTokenHelper:
    def setup_method(self):
        self.helper = TokenHelper(TiktokenCounter())

    def test_count_tokens(self):
        result = self.helper.count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty(self):
        assert self.helper.count("") == 0

    def test_count_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = self.helper.count_messages(messages)
        assert result > 0

    def test_count_messages_includes_overhead(self):
        messages = [{"role": "user", "content": "hi"}]
        msg_tokens = self.helper.count_messages(messages)
        content_tokens = self.helper.count("hi")
        assert msg_tokens > content_tokens

    def test_truncate_to_budget(self):
        long_text = "word " * 200
        truncated = self.helper.truncate(long_text, 10)
        assert self.helper.count(truncated) <= 10

    def test_truncate_short_text_unchanged(self):
        short = "hi"
        assert self.helper.truncate(short, 100) == short

    def test_fits_in_budget_true(self):
        assert self.helper.fits("short text", 100) is True

    def test_fits_in_budget_false(self):
        long_text = "word " * 200
        assert self.helper.fits(long_text, 5) is False

    def test_fit_messages_to_budget(self):
        """Should keep most recent messages that fit within budget."""
        messages = [
            {"role": "user", "content": f"message number {i} with some content"} for i in range(20)
        ]
        fitted = self.helper.fit_messages_to_budget(messages, budget=50)
        assert len(fitted) < len(messages)
        assert len(fitted) > 0
        # Should keep the most recent (last) messages
        assert fitted[-1]["content"] == messages[-1]["content"]
        # Total tokens should be within budget
        assert self.helper.count_messages(fitted) <= 50

    def test_fit_messages_all_fit(self):
        messages = [{"role": "user", "content": "hi"}]
        fitted = self.helper.fit_messages_to_budget(messages, budget=1000)
        assert len(fitted) == 1

    def test_fit_messages_empty(self):
        fitted = self.helper.fit_messages_to_budget([], budget=100)
        assert fitted == []

    def test_fit_messages_preserves_order(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        fitted = self.helper.fit_messages_to_budget(messages, budget=1000)
        assert [m["content"] for m in fitted] == ["first", "second", "third"]
