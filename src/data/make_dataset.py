# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import codecs
import re
import pathlib

RAW_DATA_DIR = "../../data/raw/data_combined/"
RAW_TRAIN = RAW_DATA_DIR+"train/"
RAW_DEV = RAW_DATA_DIR+"dev/"
RAW_TEST = RAW_DATA_DIR+"test/"

PROCESSED_DATA_DIR = "../../data/processed/"
PROCESSED_TRAIN = PROCESSED_DATA_DIR+"train/"
PROCESSED_DEV = PROCESSED_DATA_DIR+"dev/"
PROCESSED_TEST = PROCESSED_DATA_DIR+"test/"

def writeTextToFile(text, path, fileName):
    try:
        os.makedirs(path, exist_ok=True)
        f = codecs.open(path+fileName, "w", "utf-8")
        f.write(text)
        f.close()
    except:
        print("failed to write: "+fileName)

def readTextFromFile(fileName):
    try:
        f = open(fileName, 'r')
        text = f.read()
        return text
    except:
        print("failed to read: " + fileName)
        return ""

def rmWhiteSpaceFromFile(filename):
    text = readTextFromFile(filename)
    #print("-----------------------------")
    #print("BEFORE: " + text)
    text = re.sub('[\r\n\t]', '', text)
    #print("AFTER: " + text)
    return text

def rmWhiteSpaceFromFiles(raw_dir, processed_dir):
    print("processing raw files: "+raw_dir)
    print("writing to: "+processed_dir)
    rawPosDir = raw_dir + "pos/"
    rawNegDir = raw_dir + "neg/"
    processedPosDir = processed_dir + "pos/"
    processedNegDir = processed_dir + "neg/"

    posCount = 0
    for fileName in os.listdir(rawPosDir):
        text = rmWhiteSpaceFromFile(rawPosDir+fileName)
        writeTextToFile(text, processedPosDir, fileName)
        posCount = posCount+1

    negCount = 0
    for fileName in os.listdir(rawNegDir):
        text = rmWhiteSpaceFromFile(rawNegDir+fileName)
        writeTextToFile(text, processedNegDir, fileName)
        negCount = negCount+1
    print("posCount: " + str(posCount))
    print("negCount: " + str(negCount))

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    testFiles = rmWhiteSpaceFromFiles(RAW_TEST, PROCESSED_TEST)
    devFiles = rmWhiteSpaceFromFiles(RAW_DEV, PROCESSED_DEV)
    trainFiles = rmWhiteSpaceFromFiles(RAW_TRAIN, PROCESSED_TRAIN)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
