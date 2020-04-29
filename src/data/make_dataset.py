# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import shutil

import os
import codecs
import re
import pathlib
import tarfile
import os.path

# raw data dirs
RAW_DATA_DIR = "./data/raw/"
RAW_SCRAPE_DATA_DIR = RAW_DATA_DIR + "data/"
RAW_ZOMBIE_SCRAPE_DATA_DIR = RAW_DATA_DIR + "data_zombie/"

# processed data dirs
PROCESSED_DATA_DIR = "./data/processed/"
PROCESSED_SCRAPE_DATA_DIR = PROCESSED_DATA_DIR + "data/"
PROCESSED_ZOMBIE_SCRAPE_DATA_DIR = PROCESSED_DATA_DIR + "data_zombie/"

DATA_COMBINED = "./data/processed/data_combined/"

def deleteDir(path):
    try:
        shutil.rmtree(path)
    except:
        print("error deleting "+path)

def deleteFile(filePath):
    try:
        os.remove(filePath)
        print("rm file: "+file_path)
    except:
        print("maybe error removing: " + filePath)

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



def excludeTokensFromFile(filename):

    excludedTokens = [ "(Reuters)",
                       "copyright Reuters",
                       "Reuters Image",
                       "REUTERS/",
                       "Vox"
                        ]

    text = readTextFromFile(filename)
    print("-----------------------------")
    print("stripping undesired tokens from: "+filename)
    #print("BEFORE: " + text)

    for token in excludedTokens:
        text = re.sub('\stoken\s', '', text)

    text = re.sub('[\r\n\t]', '', text)
    #print("AFTER: " + text)
    return text

def excludeTokensFromFiles(raw_dir, processed_dir):
    print("processing raw files: "+raw_dir)
    print("writing to: "+processed_dir)
    rawPosDir = raw_dir + "pos/"
    rawNegDir = raw_dir + "neg/"
    processedPosDir = processed_dir + "pos/"
    processedNegDir = processed_dir + "neg/"

    posCount = 0
    for fileName in os.listdir(rawPosDir):
        text = excludeTokensFromFile(rawPosDir+fileName)
        writeTextToFile(text, processedPosDir, fileName)
        posCount = posCount+1

    negCount = 0
    for fileName in os.listdir(rawNegDir):
        text = excludeTokensFromFile(rawNegDir+fileName)
        writeTextToFile(text, processedNegDir, fileName)
        negCount = negCount+1
    print("processed posCount: " + str(posCount))
    print("processed negCount: " + str(negCount))

def excludeUndesiredTokens():
    excludeTokensFromFiles(RAW_SCRAPE_DATA_DIR, PROCESSED_SCRAPE_DATA_DIR)
    excludeTokensFromFiles(RAW_ZOMBIE_SCRAPE_DATA_DIR, PROCESSED_ZOMBIE_SCRAPE_DATA_DIR)

def generateDataCombinedTar():
    os.makedirs(DATA_COMBINED+"pos/", exist_ok=True)
    os.makedirs(DATA_COMBINED+"neg/", exist_ok=True)

    def copyAllFiles(src, dest):
        for fileName in os.listdir(src):
            try:
                shutil.copy(src+fileName, dest+fileName)
            except:
                print("failed to copy: "+fileName)

    copyAllFiles(PROCESSED_SCRAPE_DATA_DIR+"pos/", DATA_COMBINED+"pos/")
    copyAllFiles(PROCESSED_SCRAPE_DATA_DIR+"neg/", DATA_COMBINED+"neg/")
    copyAllFiles(PROCESSED_ZOMBIE_SCRAPE_DATA_DIR+"neg/", DATA_COMBINED+"neg/")

    posFileCount = len([name for name in os.listdir(DATA_COMBINED+"pos/") if os.path.isfile(os.path.join(DATA_COMBINED+"pos/", name))])
    negFileCount = len([name for name in os.listdir(DATA_COMBINED+"neg/") if os.path.isfile(os.path.join(DATA_COMBINED+"neg/", name))])
    print("posFileCount: " + str(posFileCount))
    print("negFileCount: " + str(negFileCount))

    make_tarfile(PROCESSED_DATA_DIR+"data_combined.tar.gz", DATA_COMBINED.rstrip("/") )
    print("finished tar")

def generateDataTar():
    make_tarfile(PROCESSED_DATA_DIR+"data.tar.gz", PROCESSED_SCRAPE_DATA_DIR.rstrip("/") )

def generateDataZombieTar():
    make_tarfile(PROCESSED_DATA_DIR+"data_zombie.tar.gz", PROCESSED_ZOMBIE_SCRAPE_DATA_DIR.rstrip("/"))

def make_tarfile(output_filename, source_dir):
    print("tar input dir: " + source_dir)
    print("tar output file: " + output_filename)

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # clean last run
    deleteDir(PROCESSED_SCRAPE_DATA_DIR)
    deleteDir(PROCESSED_ZOMBIE_SCRAPE_DATA_DIR)
    deleteDir(DATA_COMBINED)
    deleteDir(PROCESSED_DATA_DIR+"data.tar.gz")

    # processing
    excludeUndesiredTokens()

    generateDataTar()
    generateDataZombieTar()
    generateDataCombinedTar()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
