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
    recaps: list[dict] | None = None,
) -> str:
    """Format existing STM entities/relationships/recaps for the extraction prompt."""
    if not entities and not relationships and not recaps:
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

    if recaps:
        parts.append("Previous summaries:")
        for rc in recaps:
            parts.append(f"  - {rc['recap_text']}")

    return "\n".join(parts)


# ── Prompt builder ──────────────────────────────────────


def build_extraction_prompt(
    messages: list[dict[str, str]],
    *,
    existing_entities: list[dict] | None = None,
    existing_relationships: list[dict] | None = None,
    existing_recaps: list[dict] | None = None,
) -> str:
    """Build extraction prompt with optional existing memory context."""

    conversation = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )

    # Build existing memory section
    memory_section = ""
    if existing_entities or existing_relationships or existing_recaps:
        memory_block = build_existing_memory_block(
            existing_entities or [], existing_relationships or [],
            existing_recaps or [],
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

Output strict JSON with one or more entities, relationships, and a recap:
{{
    "entities": [
        {{
            "type": "person",
            "subtype": "colleague",
            "canonical_name": "Sarah",
            "attributes": {{"role": "manager", "department": "engineering"}},
            "attribute_confidence": {{"role": 0.90, "department": 0.80}},
            "overall_confidence": 0.88,
            "tags": ["work", "colleague"],
            "evidence": "User said 'my manager Sarah in engineering'"
        }},
        {{
            "type": "location",
            "subtype": "city",
            "canonical_name": "Seattle",
            "attributes": {{"context": "home city", "duration": "5 years"}},
            "attribute_confidence": {{"context": 0.93, "duration": 0.85}},
            "overall_confidence": 0.90,
            "tags": ["location", "home"],
            "evidence": "User said 'I've lived in Seattle for 5 years'"
        }},
        {{
            "type": "event",
            "subtype": "planned",
            "canonical_name": "Japan Trip",
            "attributes": {{"when": "next summer", "who": "with partner"}},
            "attribute_confidence": {{"when": 0.85, "who": 0.80}},
            "overall_confidence": 0.82,
            "tags": ["travel", "plans"],
            "evidence": "User mentioned planning a trip to Japan next summer"
        }}
    ],
    "relationships": [
        {{
            "subject": "user",
            "predicate": "works_with",
            "object_name": "Sarah",
            "confidence": 0.88,
            "evidence": "User said 'my manager Sarah'"
        }},
        {{
            "subject": "user",
            "predicate": "lives_in",
            "object_name": "Seattle",
            "confidence": 0.93,
            "evidence": "User said 'I've lived in Seattle'"
        }}
    ],
    "recap_text": "User works in engineering under manager Sarah. Lives in Seattle for 5 years. Planning a trip to Japan next summer with their partner.",
    "keywords": ["Sarah", "manager", "engineering", "Seattle", "Japan", "trip", "partner"],
    "tags": ["work", "location", "travel"]
}}

CONFIDENCE SCORING (CRITICAL - Follow Exactly):

NEVER use 1.0 confidence. Maximum is 0.95.

0.90-0.95: DIRECT QUOTE with zero ambiguity
  "I live in Seattle" → lives_in=Seattle (0.93)
  "My name is Alex" → name=Alex (0.95)

0.75-0.89: EXPLICITLY STATED but paraphrased
  "I work at Google" → employer=Google (0.85)
  "Sarah is my manager" → relationship=manager (0.88)

0.60-0.74: STRONGLY IMPLIED by context
  "I need to pick up the kids" (implies has children) → has_children=True (0.70)
  "The usual coffee shop" (implies regular habit) → frequents=coffee_shop (0.65)

0.45-0.59: WEAKLY IMPLIED or uncertain
  "Maybe I'll switch jobs" → considering_job_change=True (0.55)

0.30-0.44: GUESS or highly uncertain — use sparingly

ENTITY TYPES (use ONLY these):
  person: People the user knows (friend, family, partner, colleague, neighbor)
  location: Places (city, country, workplace, school, restaurant, neighborhood)
  object: Things (car, phone, book, instrument, tool, possession)
  event: Occurrences past or planned (trip, wedding, meeting, birthday, project)
  pet: Animals (dog, cat, bird, fish, horse, etc.)
  user: The user themselves (their own preferences, traits, occupation, habits)
  other: Anything else notable (organization, hobby, skill, interest)

ENTITY SUBTYPES (be specific):
  person → friend, family, partner, spouse, child, parent, sibling, colleague, manager, neighbor, teacher
  location → city, country, workplace, school, home, restaurant, park, gym, neighborhood, store
  object → vehicle, device, instrument, book, tool, clothing, furniture, food
  event → trip, wedding, birthday, meeting, project, deadline, appointment, holiday, concert
  pet → dog, cat, bird, fish, rabbit, horse, reptile
  user → self

ENTITY RULES:
- Create ONE entity per distinct thing mentioned
- Use canonical_name for primary identification (proper name or descriptive label)
- Put ALL information as attributes in the attributes dict
- Track confidence per attribute in attribute_confidence
- Use relationships to connect entities to each other and to the user
- DO NOT create entities for the assistant/AI character
- DO NOT use pronouns as canonical_name (never "he", "she", "it", "they")
- Canonical names must be at least 2 characters

RELATIONSHIP PREDICATES (use descriptive verbs):
  works_at, works_with, manages, reports_to
  lives_in, lives_with, moved_to, visits
  owns, drives, plays, studies, practices
  friends_with, married_to, parent_of, sibling_of, dating
  likes, dislikes, prefers, interested_in
  plans_to, attending, organizing, participating_in

WHAT TO EXTRACT:
- People in the user's life: family, friends, colleagues, partners
- Places: where they live, work, travel, frequent
- Things they own, use, or care about
- Events: past experiences, upcoming plans, milestones
- Pets and animals
- The user's own traits: occupation, hobbies, interests, preferences, habits

WHAT NOT TO EXTRACT:
- Facts about the ASSISTANT/AI (ignore assistant's self-descriptions)
- Assistant's opinions, reactions, or responses
- Generic small talk with no factual content

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

    # Strip markdown code blocks (closed or truncated)
    cleaned = raw.strip()
    # Try closed code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Try truncated code block (no closing ```)
        match = re.search(r"```(?:json)?\s*\n?(.*)", cleaned, re.DOTALL)
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
        # Try to repair truncated JSON by closing brackets
        repaired = _try_repair_json(cleaned)
        if repaired is not None:
            data = repaired
        else:
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


def _try_repair_json(text: str) -> dict | None:
    """Try to repair truncated JSON by closing open brackets/braces."""
    if not text or not text.startswith("{"):
        return None

    # Try progressively closing brackets
    for suffix in ["}", "]}", "]}}", "]}]}", '"}]}'  , '"}}', '"]}}']:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    # Try truncating to last complete entity in the entities array
    # Find last complete object boundary
    last_brace = text.rfind("}")
    if last_brace > 0:
        truncated = text[:last_brace + 1]
        # Close the arrays and outer object
        for suffix in ["]}", "]}",  "]}", '"]}']:
            try:
                return json.loads(truncated + suffix)
            except json.JSONDecodeError:
                continue

    return None


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 0.95] — never allow 1.0."""
    if not isinstance(value, (int, float)):
        return 0.5
    return max(0.0, min(0.95, float(value)))
