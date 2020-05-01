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

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  numFails = 0
  numSuccess = 0
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      try:
        sentence = f.read()
        sentiment = int(re.match(".*_(\d+)", file_path).group(1))
      except:
        numFails = numFails + 1
        print("failed to decode sentence")

      data["sentence"].append(sentence)
      #data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
      data["sentiment"].append(sentiment)
      numSuccess = numSuccess + 1

  print(directory+" success/fails: "+str(numSuccess)+"/"+str(numFails))
  print("data[sentence]: " + str(len(data["sentence"])))
  print("data[sentiment]: " + str(len(data["sentiment"])))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(datadir, force_download=False):
  df = load_dataset(os.path.join(datadir))
  print(df.head())
  return df

def main(train_data_dir):
    logger.info('training on '+train_data_dir)
    logger.info("TF version: " + tf.__version__)
    logger.info("Hub version: " + hub.__version__)
    logger.info ("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU'))))

    allData = download_and_load_datasets(train_data_dir)
    shuffledData = allData.sample(10000)


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

    main(sys.argv[1])
