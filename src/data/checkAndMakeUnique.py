import os

DATA_DIR = "../../data/raw/data_combined/"
TRAIN = DATA_DIR+"train/"
DEV = DATA_DIR+"dev/"
TEST = DATA_DIR+"test/"

def loadDirNamesToMemory(dataDir):
    print("loading files from: "+dataDir)
    posDir = dataDir + "pos/"
    negDir = dataDir + "neg/"

    result = set([])

    posCount = 0
    for file_path in os.listdir(posDir):
        result.add(file_path)
        posCount = posCount+1

    negCount = 0
    for file_path in os.listdir(negDir):
        result.add(file_path)
        negCount = negCount+1

    assert(posCount + negCount == len(result))
    print("posCount: " + str(posCount))
    print("negCount: " + str(negCount))
    print("result: " + str(len(result)))

    return result

def deleteFile(filePath):
    try:
        os.remove(filePath)
        print("rm file: "+file_path)
    except:
        print("maybe error removing: " + filePath)

testFiles = loadDirNamesToMemory(TEST)
devFiles = loadDirNamesToMemory(DEV)
trainFiles = loadDirNamesToMemory(TRAIN)

# make sure no duplicateFiles in TRAIN in (TEST and DEV)
print("checking and removing duplicate files in TRAIN from TEST and DEV")
dupCount = 0
while (len(trainFiles) != 0):
    trainFileToCheck = trainFiles.pop()

    if(trainFileToCheck in devFiles):
        print("dupe found in DEV and removing from TEST & DEV!!!")
        deleteFile(DEV+"/neg/"+trainFileToCheck)
        deleteFile(DEV+"/pos/"+trainFileToCheck)
        deleteFile(TEST+"/neg/"+trainFileToCheck)
        deleteFile(TEST+"/pos/"+trainFileToCheck)
        dupCount = dupCount+1

    if(trainFileToCheck in testFiles):
        print("duplicate found and removing DEV & TEST!!!")
        deleteFile(DEV+"/neg/"+trainFileToCheck)
        deleteFile(DEV+"/pos/"+trainFileToCheck)
        deleteFile(TEST+"/neg/"+trainFileToCheck)
        deleteFile(TEST+"/pos/"+trainFileToCheck)
        dupCount = dupCount+1
print("  duplicates found: "+str(dupCount))

# make sure no duplicateFiles in DEV and TEST
print("checking and removing duplicate files between dev and test")
duplicateFiles = 0
while (len(devFiles) != 0):
    devFileToCheck = devFiles.pop()
    if(devFileToCheck in testFiles):
        print("duplicate found")
        deleteFile(TEST+"pos/"+devFileToCheck)
        deleteFile(TEST+"neg/"+devFileToCheck)
        duplicateFiles = duplicateFiles + 1
print("  duplicate found: "+str(duplicateFiles))
