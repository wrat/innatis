"""
First run

```sh
python -m rasa_nlu.train -c sample_configs/config_composite_entities.yml \
        --data data/examples/dialogflow -o models \
        --fixed_model_name df-agent --project current --verbose
```

to train a model on the sample data

"""
import json
import pytest
from innatis.featurizers import UniversalSentenceEncoderFeaturizer
from rasa_nlu.model import Interpreter

def test_it_instantiates():
    assert UniversalSentenceEncoderFeaturizer({}) is not None

@pytest.mark.slow
def test_use_featurizer():
    interpreter = Interpreter.load("./models/current/df-agent")
    message = u'I will like some rice and chicken'

    result = interpreter.parse(message)
    print(json.dumps(result, indent=2))

    # todo
    # actually test something

