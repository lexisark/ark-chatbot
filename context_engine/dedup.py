"""Entity deduplication and fuzzy matching utilities."""

from __future__ import annotations

import re


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for comparison."""
    name = name.strip().lower()
    # Strip punctuation
    name = re.sub(r"[^\w\s]", "", name)
    # Strip leading articles
    name = re.sub(r"^(the|a|an)\s+", "", name)
    return name.strip()


def fuzzy_match_entity(name_a: str, name_b: str) -> bool:
    """Check if two entity names refer to the same entity.

    Matches on:
    - Exact (case-insensitive)
    - One name is a prefix/substring of the other (Max/Maxi, Bob/Bobby)
    """
    norm_a = normalize_entity_name(name_a)
    norm_b = normalize_entity_name(name_b)

    if not norm_a or not norm_b:
        return False

    # Exact match
    if norm_a == norm_b:
        return True

    # Substring/prefix match (for nicknames: Max/Maxi, Bob/Bobby)
    if norm_a in norm_b or norm_b in norm_a:
        return True

    return False


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate extracted entities by type + fuzzy name matching.

    When duplicates are found:
    - Keep the highest-confidence version's canonical_name
    - Merge attributes (later overrides earlier)
    - Keep highest confidence score
    """
    if not entities:
        return []

    # Group by entity type
    by_type: dict[str, list[dict]] = {}
    for e in entities:
        etype = e.get("type", "other")
        by_type.setdefault(etype, []).append(e)

    result = []
    for etype, type_entities in by_type.items():
        merged = _merge_group(type_entities)
        result.extend(merged)

    return result


def _merge_group(entities: list[dict]) -> list[dict]:
    """Merge entities within the same type group using fuzzy matching."""
    clusters: list[dict] = []

    for entity in entities:
        name = entity.get("canonical_name", "")
        merged = False

        for cluster in clusters:
            if fuzzy_match_entity(name, cluster["canonical_name"]):
                # Merge into existing cluster
                _merge_into(cluster, entity)
                merged = True
                break

        if not merged:
            clusters.append(dict(entity))

    return clusters


def _merge_into(target: dict, source: dict) -> None:
    """Merge source entity into target, keeping best data."""
    # Merge attributes (source overrides target)
    target_attrs = target.get("attributes", {})
    source_attrs = source.get("attributes", {})
    target["attributes"] = {**target_attrs, **source_attrs}

    # Merge attribute_confidence
    target_ac = target.get("attribute_confidence", {})
    source_ac = source.get("attribute_confidence", {})
    target["attribute_confidence"] = {**target_ac, **source_ac}

    # Keep higher confidence
    source_conf = source.get("overall_confidence", 0.0)
    target_conf = target.get("overall_confidence", 0.0)
    if source_conf > target_conf:
        target["overall_confidence"] = source_conf
        # Also adopt the higher-confidence name (proper casing)
        target["canonical_name"] = source.get("canonical_name", target["canonical_name"])

    # Merge tags
    target_tags = set(target.get("tags", []))
    source_tags = set(source.get("tags", []))
    target["tags"] = list(target_tags | source_tags)

    # Keep subtype if target doesn't have one
    if not target.get("subtype") and source.get("subtype"):
        target["subtype"] = source["subtype"]
