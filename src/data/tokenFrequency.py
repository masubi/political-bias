import os
import codecs
import re
import pathlib
import tarfile
import os.path

def readTextFromFile(fileName):
    try:
        f = open(fileName, 'r')
        text = f.read()
        return text
    except:
        print("failed to read: " + fileName)
        return ""



def iterateThroughDirectory(directory):
    print("--------------------------")
    print("sorting:  "+directory)
    print("--------------------------")
    wordToCount = {}
    for fileName in os.listdir(directory):
        text = readTextFromFile(directory+fileName)
        text.split(" ")
        for atoken in text.split():
            token=atoken.lower()
            if token in wordToCount:
                count = wordToCount.get(token)+1
                wordToCount
                count = wordToCount.get(token)+1
                wordToCount[token] = count
            else:
                wordToCount[token]=1
    import operator
    sorted_wordToCount = sorted(wordToCount.items(), key=operator.itemgetter(1))
    for (word, count) in sorted_wordToCount:
        print(word + ":" + str(count))

POSDIR = "../../data/processed/data_combined/pos/"
NEGDIR = "../../data/processed/data_combined/neg/"

def main():
    #iterateThroughDirectory(POSDIR)
    iterateThroughDirectory(NEGDIR)

if __name__ == '__main__':
    main()
