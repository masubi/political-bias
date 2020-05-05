# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import shutil
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
import math
from tensorflow import keras
import os
import re
from Optimizers import AdamWeightDecayOptimizer

import truthsayer as ts
from truthsayer import InputExample
from truthsayer import PaddingInputExample
from truthsayer import InputFeatures

#OUTPUT_DIR = 'bert-sentiment'#@param {type:"string"}
OUTPUT_DIR = '../../models/bert-sentiment'#@param {type:"string"}
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 500
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
WARMUP_PROPORTION = 0.1
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_train_steps = 1
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
#max_seq_length = 128  # Your choice here.
max_seq_length = 32  # Your choice here.

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

FullTokenizer = bert.bert_tokenization.FullTokenizer
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = ts.model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

def getPrediction(in_sentences):
  input_examples = [InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
  input_features = ts.convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
  predict_input_fn = ts.input_fn_builder(features=input_features, seq_length=max_seq_length, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)

  #[(print(prediction)) for sentence, prediction in zip(in_sentences, predictions)]

  return [(sentence, prediction['probabilities'], label_list[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

def main():
    logger.info("TF version: " + tf.__version__)
    logger.info("Hub version: " + hub.__version__)
    logger.info ("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))

    # first line is from unseen cnsnews.com article
    # rest are from unseen huffington post

    pred_sentences = [
      "“Coronavirus now, politics later – please,” Media Research Center President Brent Bozell writes in a Washington Times commentary urging the liberal media to turn their efforts from their obsession with attacking President Donald Trump to serving the greater good during a time of national crisis.",
      "President Donald Trump is rejecting calls to put a single military commander in charge of medical supplies for the COVID-19 pandemic.",
      "Boris Johnson is breathing without a ventilator and is in “good spirits” while being treated in intensive care for coronavirus symptoms, Downing Street has said.",
      "President Donald Trump reportedly owns a stake in a company that produces hydroxychloroquine, the anti-malaria drug he has repeatedly touted as a coronavirus treatment even though his experts say there’s no strong evidence it works. "
      ]

    predictions = getPrediction(pred_sentences)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    logger = logging.getLogger(__name__)
    logger.info('Args:' + str(sys.argv))
    main()
