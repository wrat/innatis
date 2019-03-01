from innatis.extractors import EntitySynonymMapper
from rasa_nlu.model import Metadata
import pytest

def test_string_value_entity_synonyms():
    entities = [{
        "entity": "test",
        "value": "chines",
        "start": 0,
        "end": 6
    }, {
        "entity": "test",
        "value": "chinese",
        "start": 0,
        "end": 6
    }, {
        "entity": "test",
        "value": "china",
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"chines": "chinese", "NYC": "New York City"}
    EntitySynonymMapper(synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 3
    assert entities[0]["value"] == "chinese"
    assert entities[1]["value"] == "chinese"
    assert entities[2]["value"] == "china"


def test_composites_entity_synonyms():
    entities = [{
        "entity": "test",
        "value": {
            "food": "egg",
            "location": "NYC",
        },
        "start": 0,
        "end": 6
    }, {
        "entity": "test",
        "value": {
            "food": "beans",
            "location": "New York City",
        },
        "start": 0,
        "end": 6
    }]
    ent_synonyms = {"egg": "eggs", "nyc": "New York City"}
    EntitySynonymMapper(synonyms=ent_synonyms).replace_synonyms(entities)
    assert len(entities) == 2
    assert entities[0]["value"]["food"] == "eggs"
    assert entities[0]["value"]["location"] == "New York City"
    assert entities[1]["value"]["food"] == "beans"
    assert entities[1]["value"]["location"] == "New York City"
