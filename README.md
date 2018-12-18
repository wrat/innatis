# Innatis

This is a library of custom [Rasa NLU](https://github.com/RasaHQ/rasa_nlu/) components that we (CarLabs) are building.

## What does the name mean?

_Viribus Innatis_ means "innate abilities" in Latin. It's a joke...

"Rasa" comes from "tabula rasa" - _blank slate_ in Latin. We (just Sam) thought it would be funny for this project to have the opposite name, since this is meant to be a suite of tools to fill in functionality for Rasa... that is, make it a not-blank-slate. So "Innate Abilities" -> Viribus Innatis.

## Featurizers

Currently the only implemented component is a featurizer

* `universal_sentence_encoder_featurizer` - Pulls the smaller USE model from TF HUB and embeds inputs as document vectors, and that vector gets sent downstream to be used as a feature.

### Planned usage

`$ pip install innatis`

Then add to your pipeline in `rasa_config.yml`. Example:

```yml
language: "en"

pipeline:
- name: "intent_featurizer_count_vectors"
- name: "innatis.featurizers.universal_sentence_encoder_featurizer.UniversalSentenceEncoderFeaturizer"
- name: "intent_classifier_tensorflow_embedding"
    intent_tokenization_flag: true
    intent_split_symbol: "."
    "epochs": 200
    "droprate": 0.5
- name: "nested_entity_extractor"
```

### Questions

Can the Rasa pipeline handle arbitrary components? That is, can we add `augmentors`, `recommenders`, etc.?
