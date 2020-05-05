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

import truthsayer as ts
from truthsayer import InputExample

#
# Globals
#
#OUTPUT_DIR = 'bert-sentiment'#@param {type:"string"}
OUTPUT_DIR = '../../models/bert-sentiment'#@param {type:"string"}
#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = True #@param {type:"boolean"}
#@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
USE_BUCKET = True #@param {type:"boolean"}
BUCKET = 'bert_truthsayer_test1' #@param {type:"string"}

allData = ts.download_and_load_datasets("../../data/processed/data/")
shuffledData = allData.sample(10000)

train = shuffledData[0:100]
dev = shuffledData[9001:9500]
test = shuffledData[9501:9999]

train.columns
train.head()

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'sentiment'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

"""#Data Preprocessing
We'll need to transform our data into a format BERT understands. This involves two steps. First, we create  `InputExample`'s using the constructor provided in the BERT library.

- `text_a` is the text we want to classify, which in this case, is the `Request` field in our Dataframe.
- `text_b` is used if we're training a model to understand the relationship between sentences (i.e. is `text_b` a translation of `text_a`? Is `text_b` an answer to the question asked by `text_a`?). This doesn't apply to our task, so we can leave `text_b` blank.
- `label` is the label for our example, i.e. True, False
"""


# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                         text_a = x[DATA_COLUMN],
                                                         text_b = None,
                                                         label = x[LABEL_COLUMN]), axis = 1)

dev_InputExamples = dev.apply(lambda x: InputExample(guid=None,
                                                     text_a = x[DATA_COLUMN],
                                                     text_b = None,
                                                     label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: InputExample(guid=None,
                                                       text_a = x[DATA_COLUMN],
                                                       text_b = None,
                                                       label = x[LABEL_COLUMN]), axis = 1)


#max_seq_length = 128  # Your choice here.
max_seq_length = 32  # Your choice here.
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

FullTokenizer = bert.bert_tokenization.FullTokenizer
tokenizer = FullTokenizer(vocab_file, do_lower_case)
#tokenizer.tokenize("This here's an example of using the BERT tokenizer")

# Convert our train and test features to InputFeatures that BERT understands.
train_features = ts.convert_examples_to_features(train_InputExamples, label_list, max_seq_length, tokenizer)
dev_features = ts.convert_examples_to_features(dev_InputExamples, label_list, max_seq_length, tokenizer)
test_features = ts.convert_examples_to_features(test_InputExamples, label_list, max_seq_length, tokenizer)


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
#NUM_TRAIN_EPOCHS = 3.0
NUM_TRAIN_EPOCHS = 6.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
print("len(train_features): "+str(len(train_features)))
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

print("num_train_steps: "+str(num_train_steps))
print("num_warmup_steps: " +str(num_warmup_steps))

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


# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = ts.input_fn_builder(
    features=train_features,
    seq_length=max_seq_length,
    is_training=True,
    drop_remainder=False)

print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

dev_input_fn = ts.input_fn_builder(
    features=dev_features,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=False)

estimator.evaluate(input_fn=dev_input_fn, steps=None)


#test_input_fn = ts.input_fn_builder(
#    features=test_features,
#    seq_length=max_seq_length,
#    is_training=False,
#    drop_remainder=False)

#estimator.evaluate(input_fn=test_input_fn, steps=None)

def main(train_data_dir):
    '''
    logger.info('training on '+train_data_dir)
    logger.info("TF version: " + tf.__version__)
    logger.info("Hub version: " + hub.__version__)
    logger.info ("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))

    allData = download_and_load_datasets(train_data_dir)
    shuffledData = allData.sample(10000)

    train = shuffledData[0:9000]
    dev = shuffledData[9001:9500]
    test = shuffledData[9501:9999]
    '''
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
    if(len(sys.argv) < 2):
        print("not enough args here")
        exit()
    main(sys.argv[1])
