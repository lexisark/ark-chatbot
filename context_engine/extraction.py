"""Extraction prompt templates and response parsing."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Extraction data models ──────────────────────────────


@dataclass
class ExtractedEntity:
    entity_type: str
    canonical_name: str
    entity_subtype: str | None = None
    attributes: dict = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class ExtractedRelationship:
    subject: str
    predicate: str
    object_name: str
    confidence: float = 0.5


@dataclass
class ExtractedRecap:
    recap_text: str
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ExtractedData:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    recap: ExtractedRecap | None = None


# ── Prompt builder ──────────────────────────────────────


def build_extraction_prompt(messages: list[dict[str, str]]) -> str:
    """Build the extraction prompt for a batch of messages."""

    conversation = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )

    return f"""Extract structured information from this conversation segment.

CONVERSATION:
{conversation}

INSTRUCTIONS:
1. Extract all notable entities (people, pets, places, objects, events, preferences)
2. Extract relationships between entities (who owns what, who knows who, etc.)
3. Write a 2-4 sentence recap summarizing the key information exchanged
4. Generate searchable keywords

CONFIDENCE SCORING:
- 0.90-0.95: Explicitly stated facts ("My dog's name is Max")
- 0.70-0.89: Strongly implied ("I take Max to the park every morning" → Max is a dog)
- 0.50-0.69: Reasonable inference
- 0.30-0.49: Weak inference, mentioned in passing

RESPOND WITH ONLY valid JSON matching this schema:
{{
    "entities": [
        {{
            "type": "pet|person|location|object|event|preference|other",
            "subtype": "dog|cat|friend|family|city|etc",
            "canonical_name": "Max",
            "attributes": {{"breed": "Golden Retriever", "age": "3 years"}},
            "overall_confidence": 0.92
        }}
    ],
    "relationships": [
        {{
            "subject": "user",
            "predicate": "owns|loves|lives_in|works_at|knows|etc",
            "object_name": "Max",
            "confidence": 0.90
        }}
    ],
    "recap_text": "2-4 sentence summary of the conversation segment",
    "keywords": ["Max", "dog", "Golden Retriever"]
}}

Rules:
- Use canonical_name consistently (proper case, full name)
- Do NOT extract the AI assistant as an entity
- Mark the user as "user" in relationships (not by their name)
- If nothing notable was discussed, return empty arrays and a brief recap
- Return ONLY the JSON, no other text"""


# ── Response parser ─────────────────────────────────────


def parse_extraction_response(raw: str) -> ExtractedData:
    """Parse LLM extraction output into structured data."""

    # Strip markdown code blocks if present
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse extraction response as JSON: {raw[:200]}")
        return ExtractedData()

    # Parse entities
    entities = []
    for e in data.get("entities", []):
        entities.append(ExtractedEntity(
            entity_type=e.get("type", "other"),
            canonical_name=e.get("canonical_name", ""),
            entity_subtype=e.get("subtype"),
            attributes=e.get("attributes", {}),
            confidence=e.get("overall_confidence", 0.5),
        ))

    # Parse relationships
    relationships = []
    for r in data.get("relationships", []):
        relationships.append(ExtractedRelationship(
            subject=r.get("subject", ""),
            predicate=r.get("predicate", ""),
            object_name=r.get("object_name", ""),
            confidence=r.get("confidence", 0.5),
        ))

    # Parse recap
    recap = None
    recap_text = data.get("recap_text")
    if recap_text:
        recap = ExtractedRecap(
            recap_text=recap_text,
            keywords=data.get("keywords", []),
            confidence=data.get("confidence", 0.5),
        )

    return ExtractedData(entities=entities, relationships=relationships, recap=recap)
