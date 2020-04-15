import newspaper
from newspaper import Article
from newspaper import news_pool
import os
import pathlib
import hashlib
import random
import codecs

debugMode = True
memoize = True
DATA_DIR = "../../data/raw/"
DOWNLOAD_DIR = DATA_DIR + "data_zombie/"


def log(msg):
    print(msg)

def debug(msg):
    if(debugMode):
        print(msg)

# parse zombie list to memory
# https://github.com/MassMove/AttackVectors/blob/master/LocalJournals/fake-local-journals-list.txt
def parseZombieSitesList():
    f = open("zombie_sites.txt","r")
    if f.mode == 'r':
        flines = f.readlines()
        aggList = []
        for line in flines:
            site = ("https://" + line).rstrip()
            aggList.append(site)
            debug("added to memory: " + site)
    f.close()
    return aggList

exclusion_set = set(["cnnespanol.cnn.com","arabic.cnn.com" "br.reuters.com","fr.reuters.com","es.reuters.com", "it.reuters.com","cn.reuters.com","reuters.zendesk.com","ru.reuters.com","ara.reuters.com","de.reuters.com","ar.reuters.com","mx.reuters.com","jp.reuters.com",
"cn.nytimes.com"])

all_news = parseZombieSitesList()

def getPapers(newspapers, exclusionSet):
    totalArticlesIncluded = 0
    for paper in newspapers:
        totalArticlesIncluded = totalArticlesIncluded +processNewsPaper(paper, exclusionSet)

    log("totalArticlesIncluded: "+str(totalArticlesIncluded))

def processNewsPaper(paper, exclusionSet):
    targetPaper = newspaper.build(paper, language ='en', memoize_articles=memoize)

    for category in targetPaper.category_urls():
        debug(category)

    articlesForPaperIncluded = 0
    for article in targetPaper.articles:
        # check if in exclusion list
        checkDomainName = article.url.split("/")[2]
        debug("check exclusion_set: "+ checkDomainName)
        if(checkDomainName in exclusionSet):
            log("excluding: "+article.url)
        else:
            articlesForPaperIncluded=articlesForPaperIncluded+downloadArticle(article.url, paper)

    log("---------------------------")
    log("paper: " + paper)
    log("articlesForPaperIncluded: " + str(articlesForPaperIncluded))
    log("totalPaper articles: " + str(targetPaper.size()))
    log("---------------------------")

    return articlesForPaperIncluded

def getSentiment(paper):
    sentiment = 1
    log("{paper}:{sentiment} =  " + paper + ":" + str(sentiment))
    return sentiment

def downloadArticle(url, paper):
    log("-------------------------------------")
    log("attempting downloading:" + url)
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        log("failed to download: "+url)
        return 0

    debug(article.text)
    if(len(article.text)<1000):
        log("article too short")
        return 0

    # filename = /{dataDir}/{trainOrDevOrTest}/{textHash}_{sentiment}
    textHash = hashlib.sha1(article.text.encode()).hexdigest()

    rand = random.randint(1,101)
    if(rand == 1):
        trainOrDevOrTest = "dev/"
    elif(rand==2):
        trainOrDevOrTest = "test/"
    else:
        trainOrDevOrTest="train/"

    sentiment = getSentiment(paper)
    if(sentiment>5):
        posOrNeg = "pos/"
    else:
        posOrNeg = "neg/"

    fileName = DOWNLOAD_DIR+trainOrDevOrTest+posOrNeg+str(textHash)+"_"+str(sentiment)

    log("downloading to: "+fileName)
    writeTextToFile(article.text, fileName)
    return 1

def writeTextToFile(text, fileName):
    f = codecs.open(fileName, "w", "utf-8")
    f.write(text)
    f.close()

def setupDataDirs():
    pathlib.Path(DATA_DIR+"train/"+"pos/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(DATA_DIR+"train/"+"neg/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(DATA_DIR+"dev/"+"pos/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(DATA_DIR+"dev/"+"neg/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(DATA_DIR+"test/"+"pos/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(DATA_DIR+"test/"+"neg/").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setupDataDirs()
    getPapers(all_news, exclusion_set)
