import os

DATA_DIR = "data_combined/"
TRAIN = DATA_DIR+"train/"
DEV = DATA_DIR+"dev/"
TEST = DATA_DIR+"test/"

def loadDirNamesToMemory(dataDir):
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
        print("Failed to remove: " + filePath)

testFiles = loadDirNamesToMemory(TEST)
devFiles = loadDirNamesToMemory(DEV)
trainFiles = loadDirNamesToMemory(TRAIN)

# make sure no dupes in TRAIN in TEST and DEV
dupCount = 0
while (len(trainFiles) != 0):
    fileToCheck = trainFiles.pop()

    if(fileToCheck in devFiles):
        print("dupe found and removing from TEST & DEV!!!")
        deleteFile(DEV+"/neg/"+fileToCheck)
        deleteFile(DEV+"/pos/"+fileToCheck)
        deleteFile(DEV+"/neg/"+fileToCheck)
        deleteFile(DEV+"/pos/"+fileToCheck)
        dupCount = dupCount+1

    if(fileToCheck in testFiles):
        print("duplicate found and removing DEV & TEST!!!")
        deleteFile(DEV+"/neg/"+fileToCheck)
        deleteFile(DEV+"/pos/"+fileToCheck)
        deleteFile(TEST+"/neg/"+fileToCheck)
        deleteFile(TEST+"/pos/"+fileToCheck)
        dupCount = dupCount+1

# make sure no dupes in DEV and TEST
dupes = 0
while (len(devFiles) != 0):
    fileToCheck = devFiles.pop()
    if(fileToCheck in testFiles):
        print("duplicate found")
        deleteFile(TEST+"pos/"+fileToCheck)
        deleteFile(TEST+"neg/"+fileToCheck)
        dupes = dupes + 1
print("duplicate found: "+str(dupes))



print("dupes found: " + str(dupCount))
