import io
import os
import pickle
from typing import Any, Optional, Dict, Text, List, Tuple
import numpy as np
import shutil

import tensorflow as tf
from tensorflow.contrib import predictor

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.tokenizers import Token

from innatis.classifiers.bert import optimization
from innatis.classifiers.bert import tokenization
from innatis.classifiers.bert import run_classifier


class BertIntentClassifier(Component):
    """Intent classifier using BERT.
    """

    name = "intent_classifier_bert"

    provides = ["intent", "intent_ranking"]

    defaults = {
        "batch_size": 42, # 128
        "epochs": 2,
        "learning_rate": 2e-5,
        "max_seq_length": 128, 
        "warmup_proportion": 0.1,
        "save_checkpoints_steps": 1000,
        "save_summary_steps": 500,
        "iterations_per_loop": 1000,
        "bert_model": "uncased_L-12_H-768_A-12",
        "bert_tfhub_module_handle": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        "checkpoint_dir": "./tmp/bert",
        "checkpoint_remove_before_training": True 
    }


    def _load_bert_params(self, config: Dict[Text, Any]) -> None:
        self.bert_model = config['bert_model']
        self.bert_tfhub_module_handle = config['bert_tfhub_module_handle']
        self.do_lower_case = self.bert_model.startswith("uncased") 


    def _load_train_params(self, config: Dict[Text, Any]) -> None:
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.max_seq_length = config['max_seq_length']
        self.warmup_proportion = config['warmup_proportion']
        self.save_checkpoints_steps = config['save_checkpoints_steps']
        self.save_summary_steps = config['save_summary_steps']
        self.iterations_per_loop = config['iterations_per_loop']
        self.checkpoint_dir = config['checkpoint_dir']
        self.checkpoint_remove_before_training = config['checkpoint_remove_before_training']


    def _load_params(self) -> None:
        self._load_bert_params(self.component_config)
        self._load_train_params(self.component_config)


    def __init__(self, 
                 component_config=None,
                 session: Optional['tf.Session'] = None,
                 label_list: Optional[np.ndarray] = None,
                 predict_fn: Optional['Predictor'] = None,
                ) -> None:
        super(BertIntentClassifier, self).__init__(component_config)

        tf.logging.set_verbosity(tf.logging.INFO)

        self.session = session
        self.label_list = label_list
        self.predict_fn = predict_fn

        self._load_params()
        self.tokenizer = run_classifier.create_tokenizer_from_hub_module(
            self.bert_tfhub_module_handle) 

        self.estimator = None


    def train(self, training_data, cfg, **kwargs):
        """Train this component."""

        # Clean up checkpoint
        if self.checkpoint_remove_before_training and os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir, ignore_errors=True)

        self.label_list = run_classifier.get_labels(training_data)

        run_config = tf.estimator.RunConfig(
            model_dir=self.checkpoint_dir,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps)
        
        train_examples = run_classifier.get_train_examples(training_data.training_examples)
        num_train_steps = int(len(train_examples) / self.batch_size * self.epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("Num examples = %d", len(train_examples))
        tf.logging.info("Batch size = %d", self.batch_size)
        tf.logging.info("Num steps = %d", num_train_steps)
        tf.logging.info("Num epochs = %d", self.epochs)

        model_fn = run_classifier.model_fn_builder(
            bert_tfhub_module_handle=self.bert_tfhub_module_handle,
            num_labels=len(self.label_list),
            learning_rate=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)
        
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": self.batch_size})
        
        train_features = run_classifier.convert_examples_to_features(
            train_examples, self.label_list, self.max_seq_length, self.tokenizer)

        train_input_fn = run_classifier.input_fn_builder(
            features=train_features,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True)

        # Start training
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        self.session = tf.Session()

        # Create predictor incase running evaluation
        self.predict_fn = predictor.from_estimator(self.estimator,
                                                   run_classifier.serving_input_fn_builder(self.max_seq_length))

                                                
    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input"""

        # Classifier needs this to be non empty, so we set to first label.
        message.data["intent"] = self.label_list[0] 

        predict_examples = run_classifier.get_test_examples([message])
        predict_features = run_classifier.convert_examples_to_features(
            predict_examples, self.label_list, self.max_seq_length, self.tokenizer)

        # Get first index since we are only classifying text blob at a time.
        example = predict_features[0]

        result = self.predict_fn({
            "input_ids": np.array(example.input_ids).reshape(-1, self.max_seq_length),
            "input_mask": np.array(example.input_mask).reshape(-1, self.max_seq_length),
            "label_ids": np.array(example.label_id).reshape(-1),
            "segment_ids": np.array(example.segment_ids).reshape(-1, self.max_seq_length),
        })

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        probabilities = list(result["probabilities"][0])

        with self.session.as_default():
            index = tf.argmax(probabilities, axis=0).eval()
            label = self.label_list[index]
            score = float(probabilities[index])

            intent = {"name": label, "confidence": score}
            intent_ranking = sorted([{"name": self.label_list[i], "confidence": float(score)} 
                                for i, score in enumerate(probabilities)],
                                key=lambda k: k["confidence"], reverse=True)

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)


    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        try:
            os.makedirs(model_dir)
        except OSError as e:
            # Be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise

        model_path = self.estimator.export_savedmodel(model_dir,
            run_classifier.serving_input_fn_builder(self.max_seq_length))

        with io.open(os.path.join(
                model_dir,
                self.name + "_label_list.pkl"), 'wb') as f:
            pickle.dump(self.label_list, f)

        return {"model_path": model_path.decode('UTF-8')}


    @classmethod
    def load(cls,
             model_dir: Text = None,
             model_metadata: 'Metadata' = None,
             cached_component: Optional['BertIntentClassifier'] = None,
             **kwargs: Any
             ) -> 'BertIntentClassifier':
            
        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("model_path"):
            model_path = os.path.normpath(meta.get("model_path"))

            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                predict_fn = predictor.from_saved_model(model_path)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_label_list.pkl"), 'rb') as f:
                label_list = pickle.load(f)

            return cls(
                component_config=meta,
                session=sess,
                label_list=label_list,
                predict_fn=predict_fn
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
            