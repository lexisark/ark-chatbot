"""Tests for entity dedup and fuzzy matching."""

import pytest

from context_engine.dedup import normalize_entity_name, fuzzy_match_entity, deduplicate_entities


class TestNormalizeEntityName:
    def test_strips_whitespace(self):
        assert normalize_entity_name("  Max  ") == "max"

    def test_lowercases(self):
        assert normalize_entity_name("Max") == "max"

    def test_strips_punctuation(self):
        assert normalize_entity_name("Max!") == "max"
        assert normalize_entity_name("'Max'") == "max"

    def test_strips_articles(self):
        assert normalize_entity_name("the park") == "park"
        assert normalize_entity_name("a dog") == "dog"

    def test_preserves_multi_word(self):
        assert normalize_entity_name("Golden Retriever") == "golden retriever"

    def test_empty_string(self):
        assert normalize_entity_name("") == ""


class TestFuzzyMatchEntity:
    def test_exact_match(self):
        assert fuzzy_match_entity("Max", "Max") is True

    def test_case_insensitive_match(self):
        assert fuzzy_match_entity("max", "Max") is True

    def test_whitespace_match(self):
        assert fuzzy_match_entity("  Max ", "Max") is True

    def test_substring_match(self):
        """'Max' should match 'Maxi' if one contains the other."""
        assert fuzzy_match_entity("Max", "Maxi") is True

    def test_no_match(self):
        assert fuzzy_match_entity("Max", "Alice") is False

    def test_nickname_match(self):
        """Common abbreviations/nicknames."""
        assert fuzzy_match_entity("Bob", "Bobby") is True

    def test_very_different_no_match(self):
        assert fuzzy_match_entity("Seattle", "Max") is False


class TestDeduplicateEntities:
    def test_exact_dupes_merged(self):
        entities = [
            {"type": "pet", "canonical_name": "Max", "attributes": {"breed": "Golden"}, "overall_confidence": 0.9},
            {"type": "pet", "canonical_name": "Max", "attributes": {"age": "3"}, "overall_confidence": 0.8},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0]["attributes"]["breed"] == "Golden"
        assert result[0]["attributes"]["age"] == "3"
        assert result[0]["overall_confidence"] == 0.9  # keeps higher

    def test_case_insensitive_dedup(self):
        entities = [
            {"type": "pet", "canonical_name": "max", "attributes": {}, "overall_confidence": 0.7},
            {"type": "pet", "canonical_name": "Max", "attributes": {"breed": "Golden"}, "overall_confidence": 0.9},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0]["canonical_name"] == "Max"  # keeps proper case

    def test_different_types_not_merged(self):
        """'Max' the pet and 'Max' the person should stay separate."""
        entities = [
            {"type": "pet", "canonical_name": "Max", "attributes": {}, "overall_confidence": 0.9},
            {"type": "person", "canonical_name": "Max", "attributes": {}, "overall_confidence": 0.8},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_fuzzy_match_dedup(self):
        entities = [
            {"type": "pet", "canonical_name": "Max", "attributes": {"breed": "Golden"}, "overall_confidence": 0.9},
            {"type": "pet", "canonical_name": "Maxi", "attributes": {"age": "3"}, "overall_confidence": 0.7},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 1
        assert result[0]["canonical_name"] == "Max"  # keeps higher confidence version's name

    def test_no_dupes_unchanged(self):
        entities = [
            {"type": "pet", "canonical_name": "Max", "attributes": {}, "overall_confidence": 0.9},
            {"type": "person", "canonical_name": "Alice", "attributes": {}, "overall_confidence": 0.8},
        ]
        result = deduplicate_entities(entities)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate_entities([]) == []
