"""Extraction prompt templates, response parsing, and validation.

Implements Arkadia-style extraction with:
- Existing memory context for dedup
- Per-attribute confidence scoring
- Pronoun filtering and canonical name validation
- Tags and evidence fields
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pronouns to reject as entity canonical names
_PRONOUNS = frozenset(
    "he she they it him her them his hers theirs its i you we me my our your".split()
)

# Names to reject (assistant/system entities)
_REJECTED_NAMES = frozenset(
    "assistant ai bot chatbot system".split()
)


# ── Data models ─────────────────────────────────────────


@dataclass
class ExtractedEntity:
    entity_type: str
    canonical_name: str
    entity_subtype: str | None = None
    attributes: dict = field(default_factory=dict)
    attribute_confidence: dict = field(default_factory=dict)
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    evidence: str | None = None


@dataclass
class ExtractedRelationship:
    subject: str
    predicate: str
    object_name: str
    confidence: float = 0.5
    evidence: str | None = None


@dataclass
class ExtractedRecap:
    recap_text: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ExtractedData:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    recap: ExtractedRecap | None = None


# ── Existing memory block builder ───────────────────────


def build_existing_memory_block(
    entities: list[dict],
    relationships: list[dict],
) -> str:
    """Format existing STM entities/relationships for the extraction prompt."""
    if not entities and not relationships:
        return ""

    parts = []

    if entities:
        parts.append("Entities:")
        for e in entities:
            name = e["canonical_name"]
            etype = e["entity_type"]
            subtype = e.get("entity_subtype", "")
            type_str = f"{etype}/{subtype}" if subtype else etype
            confidence = e.get("confidence", 0.5)

            attrs = e.get("attributes", {})
            if attrs:
                attrs_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
                parts.append(f"  - {name} ({type_str}): {attrs_str} [confidence: {confidence:.2f}]")
            else:
                parts.append(f"  - {name} ({type_str}) [confidence: {confidence:.2f}]")

    if relationships:
        parts.append("Relationships:")
        for r in relationships:
            subj = r["subject"]
            pred = r["predicate"]
            obj = r["object_name"]
            conf = r.get("confidence", 0.5)
            parts.append(f"  - {subj} → {pred} → {obj} [confidence: {conf:.2f}]")

    return "\n".join(parts)


# ── Prompt builder ──────────────────────────────────────


def build_extraction_prompt(
    messages: list[dict[str, str]],
    *,
    existing_entities: list[dict] | None = None,
    existing_relationships: list[dict] | None = None,
) -> str:
    """Build extraction prompt with optional existing memory context."""

    conversation = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )

    # Build existing memory section
    memory_section = ""
    if existing_entities or existing_relationships:
        memory_block = build_existing_memory_block(
            existing_entities or [], existing_relationships or [],
        )
        memory_section = f"""
## EXISTING MEMORY FOR THIS CONVERSATION

You have already extracted the following from previous messages in this chat:

{memory_block}

RULES FOR EXISTING MEMORY:
- If the new message refers to an existing entity (by name or pronoun), recognize it as the SAME entity
- Use the EXACT canonical_name from existing entities (e.g., if "Max" exists, use "Max" not "max")
- DO NOT create entities with pronouns as canonical names (never "he", "she", "it")
- If a relationship already exists, don't re-create it — only create NEW relationships
- Add NEW attributes to existing entities rather than duplicating entities
- Re-extract entities that have NEW or CHANGED information to reinforce memory

"""

    return f"""Extract structured information from this conversation segment.

{memory_section}## CONVERSATION TO EXTRACT FROM

{conversation}

---

Extract information that helps remember the USER better in future conversations.

Output strict JSON:
{{
    "entities": [
        {{
            "type": "pet|person|location|object|event|user|other",
            "subtype": "dog|cat|friend|family|city|workplace|etc",
            "canonical_name": "Max",
            "attributes": {{"breed": "Golden Retriever", "age": "3 years"}},
            "attribute_confidence": {{"breed": 0.95, "age": 0.80}},
            "overall_confidence": 0.92,
            "tags": ["pet", "dog"],
            "evidence": "User said 'my golden retriever Max is 3 years old'"
        }}
    ],
    "relationships": [
        {{
            "subject": "user",
            "predicate": "owns|loves|lives_in|works_at|friends_with|etc",
            "object_name": "Max",
            "confidence": 0.90,
            "evidence": "User said 'I have a dog'"
        }}
    ],
    "recap_text": "2-4 sentence summary of key information exchanged",
    "keywords": ["Max", "golden retriever", "dog", "park"],
    "tags": ["pet_care", "introduction"]
}}

CONFIDENCE SCORING (CRITICAL - Follow Exactly):

NEVER use 1.0 confidence. Maximum is 0.95.

0.90-0.95: DIRECT QUOTE with zero ambiguity
  "My dog's name is Max" → pet_name=Max (0.95)

0.75-0.89: EXPLICITLY STATED but paraphrased
  "I have a dog named Max" → has_dog=True (0.85)

0.60-0.74: STRONGLY IMPLIED by context
  "Max loves fetch" (implies has_dog) → has_dog=True (0.70)

0.45-0.59: WEAKLY IMPLIED or uncertain
  "Maybe I'll get a dog" → wants_dog=maybe (0.55)

0.30-0.44: GUESS or highly uncertain — use sparingly

ENTITY TYPES (use ONLY these):
  user: The user themselves (preferences, characteristics, occupation)
  pet: Any animal (dog, cat, bird, fish, etc.)
  person: Other people (friend, family_member, partner, colleague)
  location: Places (city, workplace, home, landmark)
  object: Possessions (car, phone, hobby items)
  event: Significant occurrences (planned or past)
  other: Anything else important

ENTITY RULES:
- Create ONE entity per distinct thing
- Use canonical_name for primary identification
- Put ALL information as attributes in the attributes dict
- Track confidence per attribute in attribute_confidence
- Use relationships to connect entities
- DO NOT create entities for the assistant/AI character
- DO NOT use pronouns as canonical_name (never "he", "she", "it")
- Canonical names must be at least 2 characters

RELATIONSHIP RULES:
- subject: "user" OR entity canonical_name
- predicate: owns, loves, likes, lives_in, works_at, friends_with, etc.
- object_name: entity canonical_name

WHAT TO EXTRACT:
- Facts about the USER: life, family, pets, location, occupation, hobbies
- User preferences: communication style, interests, behavioral patterns
- Information from user messages and user-shared images

WHAT NOT TO EXTRACT:
- Facts about the ASSISTANT/AI (ignore assistant's self-descriptions)
- Assistant's opinions or responses as facts

REINFORCEMENT:
- Extract ALL facts mentioned in THIS conversation, even if mentioned before
- Re-extraction reinforces important information and builds confidence
- Every mention matters for memory strength

RECAP: Keep under 60 tokens. Include entity names for searchability.
KEYWORDS: 10-20 searchable terms (names, types, activities, locations).

Return ONLY the JSON, no other text."""


# ── Response parser ─────────────────────────────────────


def parse_extraction_response(raw: str) -> ExtractedData:
    """Parse LLM extraction output into structured data."""

    # Strip markdown code blocks
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    # Try finding JSON object in text
    if not cleaned.startswith("{"):
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse extraction response: {raw[:200]}")
        return ExtractedData()

    # Validate and clean
    data = validate_extraction(data)

    # Parse entities
    entities = []
    for e in data.get("entities", []):
        entities.append(ExtractedEntity(
            entity_type=e.get("type", "other"),
            canonical_name=e.get("canonical_name", ""),
            entity_subtype=e.get("subtype"),
            attributes=e.get("attributes", {}),
            attribute_confidence=e.get("attribute_confidence", {}),
            confidence=e.get("overall_confidence", 0.5),
            tags=e.get("tags", []),
            evidence=e.get("evidence"),
        ))

    # Parse relationships
    relationships = []
    for r in data.get("relationships", []):
        relationships.append(ExtractedRelationship(
            subject=r.get("subject", ""),
            predicate=r.get("predicate", ""),
            object_name=r.get("object_name", ""),
            confidence=r.get("confidence", 0.5),
            evidence=r.get("evidence"),
        ))

    # Parse recap
    recap = None
    recap_text = data.get("recap_text")
    if recap_text:
        recap = ExtractedRecap(
            recap_text=recap_text,
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            confidence=data.get("confidence", 0.5),
        )

    return ExtractedData(entities=entities, relationships=relationships, recap=recap)


# ── Validation ──────────────────────────────────────────


def validate_extraction(data: dict) -> dict:
    """Validate and clean extraction data. Rejects pronouns, clamps confidence."""

    # Validate entities
    valid_entities = []
    for e in data.get("entities", []):
        name = e.get("canonical_name", "").strip()

        # Reject pronouns
        if name.lower() in _PRONOUNS:
            logger.debug(f"Rejected pronoun entity: '{name}'")
            continue

        # Reject short names
        if len(name) < 2:
            logger.debug(f"Rejected short entity name: '{name}'")
            continue

        # Reject assistant entities
        if name.lower() in _REJECTED_NAMES:
            logger.debug(f"Rejected assistant entity: '{name}'")
            continue

        # Clamp confidence
        e["overall_confidence"] = _clamp_confidence(e.get("overall_confidence", 0.5))

        # Clamp attribute confidences
        attr_conf = e.get("attribute_confidence", {})
        for k, v in attr_conf.items():
            attr_conf[k] = _clamp_confidence(v)
        e["attribute_confidence"] = attr_conf

        valid_entities.append(e)

    data["entities"] = valid_entities

    # Validate relationships
    valid_rels = []
    for r in data.get("relationships", []):
        r["confidence"] = _clamp_confidence(r.get("confidence", 0.5))
        valid_rels.append(r)

    data["relationships"] = valid_rels

    return data


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 0.95] — never allow 1.0."""
    if not isinstance(value, (int, float)):
        return 0.5
    return max(0.0, min(0.95, float(value)))
