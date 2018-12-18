# Innatis

This is a library of custom [Rasa NLU](https://github.com/RasaHQ/rasa_nlu/) components that we (CarLabs) are building.

## What does the name mean?

_Viribus Innatis_ means "innate abilities" in Latin. It's a joke...

"Rasa" comes from "tabula rasa" - _blank slate_ in Latin. We (just Sam) thought it would be funny for this project to have the opposite name, since this is meant to be a suite of tools to fill in functionality for Rasa... that is, make it a not-blank-slate. So "Innate Abilities" -> Viribus Innatis.

## Usage

`$ pip install innatis`

Then add to your pipeline in your `rasa_config.yml`. Example pipeline can be found in `sample_rasa_innatis_config.yml`.

## Components

### Extractors

* `composite_entity_extractor` - Given entities extracted by another extractor (`ner_crf` seems to be the best for now), splits them into composite entities, similar to [DialogFlow](https://dialogflow.com/docs/entities/developer-entities#developer_composite).

### Featurizers

* `universal_sentence_encoder_featurizer` - Pulls the smaller USE model from TF HUB and embeds inputs as document vectors, and that vector gets sent downstream to be used as a feature.

## Development

```sh
git clone git@github.com:Revmaker/innatis.git
cd innatis
pipenv install --skip-lock
# you don't _need_ to skip-lock, if you are immortal and patient https://github.com/pypa/pipenv/issues/2284#issuecomment-397867839

# do some stuff
# write some tests

pipenv run python test.py
```
