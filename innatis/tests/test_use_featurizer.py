import json
import pytest
from innatis.featurizers import UniversalSentenceEncoderFeaturizer
from rasa.nlu.model import Interpreter
from rasa.nlu import config, train

@pytest.mark.slow
def test_it_instantiates():
    assert UniversalSentenceEncoderFeaturizer({}) is not None

@pytest.mark.slow
def test_train_featurizer():
    (trained, _, _) = train.do_train(
        config.load('sample_configs/sample_use_featurizer.yml'),
        data='data/examples/dialogflow',
        path='models',
        project='current',
        fixed_model_name='use-featurizer')

    assert trained.pipeline

@pytest.mark.slow
def test_use_featurizer():
    interpreter = Interpreter.load("./models/current/use-featurizer")

    assert interpreter.pipeline
    assert interpreter.parse("hello") is not None
    assert interpreter.parse("I will like some rice and chicken") is not None
