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
    # filename = /{dataDir}/{textHash}_{sentiment}
    textHash = hashlib.sha1(article.text.encode()).hexdigest()

    sentiment = getSentiment(paper)
    if(sentiment>5):
        posOrNeg = "pos/"
    else:
        posOrNeg = "neg/"

    pathName = DOWNLOAD_DIR+posOrNeg
    fileName = str(textHash)+"_"+str(sentiment)

    log("downloading to: "+pathName+fileName)
    writeTextToFile(article.text, pathName, fileName)
    return 1

def writeTextToFile(text, path, fileName):
    try:
        os.makedirs(path, exist_ok=True)
        f = codecs.open(path+fileName, "w", "utf-8")
        f.write(text)
        f.close()
    except:
        print("failed to write: "+fileName)

if __name__ == "__main__":
    getPapers(all_news, exclusion_set)
