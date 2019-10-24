from innatis.extractors import EntitySynonymMapper
from rasa.nlu.model import Metadata
import pytest

def test_string_value_match():
    entities = [{
        "entity": "cuisine",
        "value": "chinese",
        "start": 0,
        "end": 6
    }, {
        "entity": "cuisine",
        "value": "chineese",
        "start": 0,
        "end": 6
    }, {
        "entity": "cuisine",
        "value": "Italian",
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"chineese": "chinese", "nyc": "New York City"}
    EntitySynonymMapper(synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 3
    # Does not replace original value
    assert entities[0]["value"] == "chinese"
    # Replaces synonym with the original value
    assert entities[1]["value"] == "chinese"
    # Preserves case of non-matched entities
    assert entities[2]["value"] == "Italian"


def test_composite_entity_match():
    entities = [{
        "entity": "restaurant",
        "value": {
            "cuisine": "chineese",
            "location": "NYC",
        },
        "start": 0,
        "end": 6
    }, {
        "entity": "restaurant",
        "value": {
            "cuisine": "Chinese",
            "location": "New York City",
        },
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"chineese": "chinese", "nyc": "New York City"}
    EntitySynonymMapper({"fuzzy_matching": False}, synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 2
    # Replaces synonyms with original values
    assert entities[0]["value"]["cuisine"] == "chinese"
    assert entities[0]["value"]["location"] == "New York City"
    # Preserves case of non-matched entities; Does not replace original value
    assert entities[1]["value"]["cuisine"] == "Chinese"
    assert entities[1]["value"]["location"] == "New York City"

def test_string_value_fuzzy_match():
    entities = [{
        "entity": "cuisine",
        "value": "chinese",
        "start": 0,
        "end": 6
    }, {
        "entity": "cuisine",
        "value": "chinees",
        "start": 0,
        "end": 6
    }, {
        "entity": "cuisine",
        "value": "china",
        "start": 0,
        "end": 6
    }, {
        "entity": "location",
        "value": "NewYork City",
        "start": 0,
        "end": 6
    }, {
        "entity": "cuisine",
        "value": "Italian",
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"chineese": "chinese", "nyc": "New York City"}
    EntitySynonymMapper({"fuzzy_matching": True}, synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 5
    # Does not replace original value
    assert entities[0]["value"] == "chinese"
    # Replaces fuzzy-matched synonym with original value (chinees -> chineese -> chinese)
    assert entities[1]["value"] == "chinese"
    # Does not fuzzy-match if it's too fuzzy (< fuzzy_threshold similarity)
    assert entities[2]["value"] == "china"
    # Fuzzy-matches with original value (NewYork City -> New York City)
    assert entities[3]["value"] == "New York City"
    # Preserves case of non-matched entities
    assert entities[4]["value"] == "Italian"

def test_composite_entity_fuzzy_match():
    entities = [{
        "entity": "restaurant",
        "value": {
            "cuisine": "chinees",
            "location": "NYC",
        },
        "start": 0,
        "end": 6
    }, {
        "entity": "restaurant",
        "value": {
            "cuisine": "Italian",
            "location": "NewYork City",
        },
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"chineese": "chinese", "nyc": "New York City"}
    EntitySynonymMapper({"fuzzy_matching": True}, synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 2
    # Replaces fuzzy-matched synonym with original value (chinees -> chineese -> chinese)
    assert entities[0]["value"]["cuisine"] == "chinese"
    # Replaces synonym with the original value
    assert entities[0]["value"]["location"] == "New York City"
    # Preserves case of non-matched entities
    assert entities[1]["value"]["cuisine"] == "Italian"
    # Fuzzy-matches with original value (NewYork City -> New York City)
    assert entities[1]["value"]["location"] == "New York City"
