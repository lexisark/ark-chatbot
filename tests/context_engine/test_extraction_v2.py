"""Tests for v2 extraction with existing memory context and dedup."""

import json

import pytest

from context_engine.extraction import (
    build_extraction_prompt,
    parse_extraction_response,
    validate_extraction,
    build_existing_memory_block,
)


class TestBuildExtractionPromptWithMemory:
    def test_includes_existing_memory_block(self):
        messages = [{"role": "user", "content": "Max loves the park."}]
        existing_entities = [
            {"canonical_name": "Max", "entity_type": "pet", "entity_subtype": "dog",
             "attributes": {"breed": "Golden Retriever"}, "confidence": 0.92},
        ]
        existing_relationships = [
            {"subject": "user", "predicate": "owns", "object_name": "Max", "confidence": 0.90},
        ]

        prompt = build_extraction_prompt(
            messages,
            existing_entities=existing_entities,
            existing_relationships=existing_relationships,
        )

        assert "EXISTING MEMORY" in prompt
        assert "Max" in prompt
        assert "Golden Retriever" in prompt
        assert "user → owns → Max" in prompt or "owns" in prompt

    def test_without_existing_memory(self):
        messages = [{"role": "user", "content": "I have a cat."}]
        prompt = build_extraction_prompt(messages)

        assert "EXISTING MEMORY" not in prompt
        assert "entities" in prompt  # Still has schema

    def test_includes_confidence_scoring_rules(self):
        prompt = build_extraction_prompt([{"role": "user", "content": "hi"}])

        assert "0.90-0.95" in prompt
        assert "0.75-0.89" in prompt
        assert "NEVER use 1.0" in prompt or "never" in prompt.lower()

    def test_includes_entity_type_definitions(self):
        prompt = build_extraction_prompt([{"role": "user", "content": "hi"}])

        for etype in ["pet", "person", "location", "object", "event", "user"]:
            assert etype in prompt

    def test_includes_attribute_confidence_in_schema(self):
        prompt = build_extraction_prompt([{"role": "user", "content": "hi"}])
        assert "attribute_confidence" in prompt

    def test_includes_tags_in_schema(self):
        prompt = build_extraction_prompt([{"role": "user", "content": "hi"}])
        assert "tags" in prompt

    def test_includes_evidence_in_schema(self):
        prompt = build_extraction_prompt([{"role": "user", "content": "hi"}])
        assert "evidence" in prompt


class TestBuildExistingMemoryBlock:
    def test_formats_entities(self):
        entities = [
            {"canonical_name": "Max", "entity_type": "pet", "entity_subtype": "dog",
             "attributes": {"breed": "Golden Retriever", "age": "3 years"}, "confidence": 0.92},
        ]
        block = build_existing_memory_block(entities, [])
        assert "Max" in block
        assert "pet" in block
        assert "Golden Retriever" in block

    def test_formats_relationships(self):
        relationships = [
            {"subject": "user", "predicate": "owns", "object_name": "Max", "confidence": 0.90},
        ]
        block = build_existing_memory_block([], relationships)
        assert "user" in block
        assert "owns" in block
        assert "Max" in block

    def test_empty_memory(self):
        block = build_existing_memory_block([], [])
        assert block == ""


class TestParseExtractionResponseV2:
    def test_parse_with_attribute_confidence(self):
        raw = json.dumps({
            "entities": [{
                "type": "pet", "subtype": "dog", "canonical_name": "Max",
                "attributes": {"breed": "Golden Retriever", "age": "3 years"},
                "attribute_confidence": {"breed": 0.95, "age": 0.80},
                "overall_confidence": 0.92,
                "tags": ["pet", "dog"],
                "evidence": "User said 'my golden retriever Max is 3'",
            }],
            "relationships": [{
                "subject": "user", "predicate": "owns", "object_name": "Max",
                "confidence": 0.90, "evidence": "User said 'my dog'",
            }],
            "recap_text": "User discussed their golden retriever Max.",
            "keywords": ["Max", "golden retriever"],
            "tags": ["pet_care"],
        })

        result = parse_extraction_response(raw)

        assert result.entities[0].attribute_confidence["breed"] == 0.95
        assert result.entities[0].tags == ["pet", "dog"]
        assert result.entities[0].evidence == "User said 'my golden retriever Max is 3'"
        assert result.recap.tags == ["pet_care"]

    def test_parse_missing_optional_fields(self):
        raw = json.dumps({
            "entities": [{"type": "person", "canonical_name": "Alice", "overall_confidence": 0.8}],
            "relationships": [],
        })

        result = parse_extraction_response(raw)
        assert result.entities[0].attribute_confidence == {}
        assert result.entities[0].tags == []
        assert result.entities[0].evidence is None


class TestValidateExtraction:
    def test_rejects_pronoun_entities(self):
        data = {
            "entities": [
                {"type": "person", "canonical_name": "he", "overall_confidence": 0.8},
                {"type": "pet", "canonical_name": "Max", "overall_confidence": 0.9},
                {"type": "person", "canonical_name": "she", "overall_confidence": 0.7},
            ],
            "relationships": [],
        }

        cleaned = validate_extraction(data)
        names = [e["canonical_name"] for e in cleaned["entities"]]
        assert "Max" in names
        assert "he" not in names
        assert "she" not in names

    def test_rejects_short_names(self):
        data = {
            "entities": [
                {"type": "person", "canonical_name": "I", "overall_confidence": 0.8},
                {"type": "pet", "canonical_name": "Max", "overall_confidence": 0.9},
            ],
            "relationships": [],
        }

        cleaned = validate_extraction(data)
        names = [e["canonical_name"] for e in cleaned["entities"]]
        assert "Max" in names
        assert "I" not in names

    def test_clamps_confidence_to_095(self):
        data = {
            "entities": [
                {"type": "pet", "canonical_name": "Max", "overall_confidence": 1.0},
            ],
            "relationships": [
                {"subject": "user", "predicate": "owns", "object_name": "Max", "confidence": 1.0},
            ],
        }

        cleaned = validate_extraction(data)
        assert cleaned["entities"][0]["overall_confidence"] == 0.95
        assert cleaned["relationships"][0]["confidence"] == 0.95

    def test_clamps_negative_confidence(self):
        data = {
            "entities": [
                {"type": "pet", "canonical_name": "Max", "overall_confidence": -0.5},
            ],
            "relationships": [],
        }

        cleaned = validate_extraction(data)
        assert cleaned["entities"][0]["overall_confidence"] == 0.0

    def test_rejects_assistant_entities(self):
        data = {
            "entities": [
                {"type": "user", "canonical_name": "Assistant", "overall_confidence": 0.8},
                {"type": "pet", "canonical_name": "Max", "overall_confidence": 0.9},
            ],
            "relationships": [],
        }

        cleaned = validate_extraction(data)
        names = [e["canonical_name"] for e in cleaned["entities"]]
        assert "Max" in names
        assert "Assistant" not in names


class TestExtractionHandlerWithMemory:
    """Test that the extraction handler loads existing memory before calling LLM."""

    async def test_handler_passes_existing_memory(self, db_session):
        from db.queries import create_chat, create_message
        from db.models import MessageRole
        from context_engine.stm_manager import STMManager
        from worker.extraction_handler import run_batch_extraction
        from providers.base import ChatResponse

        chat = await create_chat(db_session)
        # Add existing entity
        stm = STMManager()
        await stm.upsert_entity(db_session, chat.id, "pet", "Max",
                                attributes={"breed": "Golden"}, confidence=0.9)

        # Add messages
        await create_message(db_session, chat.id, MessageRole.USER, "Max went to the park today")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "That sounds fun!")

        class MemoryAwareProvider:
            def __init__(self):
                self.received_prompt = None

            async def chat(self, messages, **kw):
                self.received_prompt = messages[0]["content"] if messages else ""
                return ChatResponse(
                    content=json.dumps({
                        "entities": [{"type": "pet", "canonical_name": "Max",
                                      "attributes": {"activity": "park"}, "overall_confidence": 0.85}],
                        "relationships": [],
                        "recap_text": "Max went to the park.",
                        "keywords": ["Max", "park"],
                    }),
                    model="test", tokens_in=100, tokens_out=50, latency_ms=10,
                )

            async def chat_stream(self, *a, **kw):
                yield

        provider = MemoryAwareProvider()
        await run_batch_extraction(db_session, provider, stm, chat.id)

        # The prompt should contain existing memory context
        assert provider.received_prompt is not None
        assert "Max" in provider.received_prompt
        assert "EXISTING MEMORY" in provider.received_prompt or "Golden" in provider.received_prompt
