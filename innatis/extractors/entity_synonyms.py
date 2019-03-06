import editdistance
import os
import warnings
from typing import Any, Dict, Optional, Text

from rasa_nlu import utils
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message, TrainingData
from rasa_nlu.utils import write_json_to_file

ENTITY_SYNONYMS_FILE_NAME = "entity_synonyms.json"


class EntitySynonymMapper(EntityExtractor):
    name = "ner_synonyms"

    provides = ["entities"]

    defaults = {
        "fuzzy_matching": False,
        "fuzzy_threshold": 0.9
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Text]] = None,
                 synonyms: Optional[Dict[Text, Any]] = None) -> None:

        super(EntitySynonymMapper, self).__init__(component_config)

        self.synonyms = synonyms if synonyms else {}

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get("entities", []):
                entity_val = example.text[entity["start"]:entity["end"]]
                self.add_entities_if_synonyms(entity_val,
                                              str(entity.get("value")))

    def process(self, message: Message, **kwargs: Any) -> None:

        updated_entities = message.get("entities", [])[:]
        self.replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:

        if self.synonyms:
            entity_synonyms_file = os.path.join(model_dir,
                                                ENTITY_SYNONYMS_FILE_NAME)
            write_json_to_file(entity_synonyms_file, self.synonyms,
                               separators=(',', ': '))
            return {"synonyms_file": ENTITY_SYNONYMS_FILE_NAME}
        else:
            return {"synonyms_file": None}

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional['EntitySynonymMapper'] = None,
             **kwargs: Any
             ) -> 'EntitySynonymMapper':

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("synonyms_file")
        if not file_name:
            synonyms = None
            return cls(meta, synonyms)

        entity_synonyms_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entity_synonyms_file):
            synonyms = utils.read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            warnings.warn("Failed to load synonyms file from '{}'"
                          "".format(entity_synonyms_file))
        return cls(meta, synonyms)

    def replace_synonyms(self, entities):
        for entity in entities:
            entity_value = entity["value"]
            if type (entity_value) is dict:
                # Needed so that we dont add the processors mutiple times
                add_processor_name = False
                for key, value in entity_value.items():
                    lookup_value = str(value).lower()
                    if lookup_value in self.synonyms:
                        entity["value"][key] = self.synonyms[lookup_value]
                        add_processor_name = True
                    elif self.component_config["fuzzy_matching"]:
                        matched = self.fuzzy_match_entity(lookup_value)
                        if matched:
                            entity["value"][key] = matched
                            add_processor_name = True
                if add_processor_name:
                    self.add_processor_name(entity)
            else:
                # need to wrap in `str` to handle e.g. entity values of type int
                lookup_value = str(entity_value).lower()
                if lookup_value in self.synonyms:
                    entity["value"] = self.synonyms[lookup_value]
                    self.add_processor_name(entity)
                elif self.component_config["fuzzy_matching"]:
                    matched = self.fuzzy_match_entity(lookup_value)
                    if matched:
                        entity["value"] = matched
                        self.add_processor_name(entity)

    def fuzzy_match_entity(self, lookup_value):
        threshold = self.component_config["fuzzy_threshold"]
        fuzzy_match = None

        # Match the synonyms
        for synonym in self.synonyms.keys():
            similarity = EntitySynonymMapper.calc_similarity(synonym, lookup_value)
            if similarity >= threshold:
                candidate = (similarity, self.synonyms[synonym])
                fuzzy_match = max(candidate, fuzzy_match) if fuzzy_match else candidate

        # Match the original values
        for original_value in set(self.synonyms.values()):
            similarity = EntitySynonymMapper.calc_similarity(original_value.lower(), lookup_value)
            if similarity >= threshold:
                candidate = (similarity, original_value)
                fuzzy_match = max(candidate, fuzzy_match) if fuzzy_match else candidate

        return fuzzy_match[1] if fuzzy_match else None

    @staticmethod
    def calc_similarity(str_a, str_b):
        """
        Returns a similarity from 0.0 to 1.0 between 2 strings

        This converts the editdistance to a percentage
        Based on the equation used here:
        https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio
        """
        distance = editdistance.eval(str_a, str_b)
        matches = max(len(str_a), len(str_b)) - distance
        return 2 * matches / (len(str_a) + len(str_b))

    def add_entities_if_synonyms(self, entity_a, entity_b):
        if entity_b is not None:
            original = str(entity_a)
            replacement = str(entity_b)

            if original != replacement:
                original = original.lower()
                if (original in self.synonyms and
                        self.synonyms[original] != replacement):
                    warnings.warn("Found conflicting synonym definitions "
                                  "for {}. Overwriting target {} with {}. "
                                  "Check your training data and remove "
                                  "conflicting synonym definitions to "
                                  "prevent this from happening."
                                  "".format(repr(original),
                                            repr(self.synonyms[original]),
                                            repr(replacement)))

                self.synonyms[original] = replacement
