import os
import codecs
import re

RAW_DATA_DIR = "../../data/raw/data_combined/"
RAW_TRAIN = RAW_DATA_DIR+"train/"
RAW_DEV = RAW_DATA_DIR+"dev/"
RAW_TEST = RAW_DATA_DIR+"test/"

PROCESSED_DATA_DIR = "../../data/processed/"
PROCESSED_TRAIN = PROCESSED_DATA_DIR+"train/"
PROCESSED_DEV = PROCESSED_DATA_DIR+"dev/"
PROCESSED_TEST = PROCESSED_DATA_DIR+"test/"

def writeTextToFile(text, fileName):
    try:
        f = codecs.open(fileName, "w", "utf-8")
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
    print("-----------------------------")
    print("BEFORE: " + text)
    text = re.sub('[\r\n\t]', '', text)
    print("AFTER: " + text)
    return text

def rmWhitespace(raw_dir, processed_dir):
    print("loading files from: "+raw_dir)
    posDir = raw_dir + "pos/"
    negDir = raw_dir + "neg/"

    posCount = 0
    for file_path in os.listdir(posDir):
        rmWhiteSpaceFromFile(posDir+file_path)
        posCount = posCount+1

    negCount = 0
    for file_path in os.listdir(negDir):
        rmWhiteSpaceFromFile(negDir+file_path)
        negCount = negCount+1

    #assert(posCount + negCount == len(result))
    print("posCount: " + str(posCount))
    print("negCount: " + str(negCount))
    #print("result: " + str(len(result)))

    #return result

testFiles = rmWhitespace(RAW_TEST, PROCESSED_TEST)
#devFiles = rmWhitespace(DEV)
#trainFiles = rmWhitespace(TRAIN)
