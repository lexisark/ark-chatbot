"""Tests for extraction prompt building and response parsing."""

import json

from context_engine.extraction import build_extraction_prompt, parse_extraction_response


class TestBuildExtractionPrompt:
    def test_produces_string(self):
        messages = [
            {"role": "user", "content": "I have a dog named Max."},
            {"role": "assistant", "content": "That's great! What breed is Max?"},
        ]
        prompt = build_extraction_prompt(messages)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_message_content(self):
        messages = [
            {"role": "user", "content": "My sister Alice lives in Seattle."},
        ]
        prompt = build_extraction_prompt(messages)
        assert "Alice" in prompt
        assert "Seattle" in prompt

    def test_includes_json_schema_hint(self):
        messages = [{"role": "user", "content": "hi"}]
        prompt = build_extraction_prompt(messages)
        assert "entities" in prompt
        assert "relationships" in prompt
        assert "recap_text" in prompt


class TestParseExtractionResponse:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "entities": [
                {
                    "type": "pet",
                    "subtype": "dog",
                    "canonical_name": "Max",
                    "attributes": {"breed": "Golden Retriever"},
                    "overall_confidence": 0.92,
                }
            ],
            "relationships": [
                {
                    "subject": "user",
                    "predicate": "owns",
                    "object_name": "Max",
                    "confidence": 0.90,
                }
            ],
            "recap_text": "User mentioned their golden retriever Max.",
            "keywords": ["Max", "dog", "Golden Retriever"],
        })

        result = parse_extraction_response(raw)

        assert len(result.entities) == 1
        assert result.entities[0].canonical_name == "Max"
        assert result.entities[0].entity_type == "pet"
        assert result.entities[0].attributes["breed"] == "Golden Retriever"
        assert result.entities[0].confidence == 0.92

        assert len(result.relationships) == 1
        assert result.relationships[0].subject == "user"
        assert result.relationships[0].predicate == "owns"
        assert result.relationships[0].object_name == "Max"

        assert result.recap.recap_text == "User mentioned their golden retriever Max."
        assert "Max" in result.recap.keywords

    def test_parse_malformed_json(self):
        result = parse_extraction_response("not valid json {{{")
        assert len(result.entities) == 0
        assert len(result.relationships) == 0
        assert result.recap is None

    def test_parse_partial_data(self):
        raw = json.dumps({
            "entities": [
                {"type": "person", "canonical_name": "Alice", "overall_confidence": 0.8}
            ],
            # Missing relationships and recap
        })

        result = parse_extraction_response(raw)
        assert len(result.entities) == 1
        assert result.entities[0].canonical_name == "Alice"
        assert len(result.relationships) == 0
        assert result.recap is None

    def test_parse_empty_json(self):
        result = parse_extraction_response("{}")
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    def test_parse_json_in_code_block(self):
        """LLMs sometimes wrap JSON in markdown code blocks."""
        raw = '```json\n{"entities": [{"type": "pet", "canonical_name": "Max", "overall_confidence": 0.9}], "relationships": [], "recap_text": "test", "keywords": ["Max"]}\n```'

        result = parse_extraction_response(raw)
        assert len(result.entities) == 1
        assert result.entities[0].canonical_name == "Max"
