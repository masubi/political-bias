import newspaper
from newspaper import Article
from newspaper import news_pool
import os
import pathlib
import hashlib
import random
import codecs

debugMode = False
memoize = True

def log(msg):
    print(msg)

def debug(msg):
    if(debugMode):
        print(msg)

real_news = [
    #"http://cnn.com",
    "http://reuters.com",
    "https://www.bbc.com",
    "https://www.npr.org",
    "https://www.nytimes.com",
    "https://www.wsj.com",
    "https://www.washingtonpost.com",
    "https://www.economist.com",
    "https://www.ap.org",
    "https://www.theatlantic.com",
    "https://gizmodo.com/",
    "https://www.wired.com/",
    "https://www.nature.com/",
    "https://www.vox.com/",
    "https://www.theguardian.com/",
    "https://www.kiro7.com/",
    "https://www.staradvertiser.com/",
    "https://www.independent.co.uk/",
    "https://www.newsweek.com/",
    "https://www.businessinsider.com/",


    # Returns 0 articles
    "https://www.forbes.com",
    "https://www.bloomberg.com",

]

fake_news = [
    "https://www.oann.com/",
    "https://www.breitbart.com",
    "https://www.infowars.com",
    "https://www.foxnews.com",
    "https://www.theepochtimes.com/",

    # Returns 0 articles
    "https://mynorthwest.com/"
]

all_news = real_news + fake_news

exclusion_set = set(["cnnespanol.cnn.com","arabic.cnn.com" "br.reuters.com","fr.reuters.com","es.reuters.com", "it.reuters.com","cn.reuters.com","reuters.zendesk.com","ru.reuters.com","ara.reuters.com","de.reuters.com","ar.reuters.com","mx.reuters.com","jp.reuters.com",
"cn.nytimes.com"])

sentimentMap = {
    "http://cnn.com" : 10,
    "http://reuters.com" : 10,
    "https://www.bbc.com" : 10,
    "https://www.npr.org" : 10,
    "https://www.nytimes.com" : 7,
    "https://www.wsj.com" : 10,
    "https://www.washingtonpost.com" : 7,
    "https://www.economist.com" : 10,
    "https://www.ap.org" : 10,
    "https://www.theatlantic.com" : 10,
    "https://gizmodo.com/" : 7,
    "https://www.wired.com/" : 7,
    "https://www.nature.com/" : 10,
    "https://www.vox.com/" : 7,
    "https://www.kiro7.com/" : 7,
    "https://www.staradvertiser.com/" : 6,
    "https://www.independent.co.uk/" : 6,
    "https://www.theguardian.com/" : 9,
    "https://www.newsweek.com/" : 5,
    "https://www.businessinsider.com/" : 5,

    "https://www.breitbart.com" : 1,
    "https://www.infowars.com" : 1,
    "https://www.foxnews.com" : 3,
    "https://mynorthwest.com/" : 1,
    "https://www.oann.com/" : 2,
    "https://www.theepochtimes.com/" : 1
}

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
    sentiment = sentimentMap[paper]
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

    filePath = DATA_DIR+posOrNeg
    fileName = str(textHash)+"_"+str(sentiment)

    log("downloading to: "+filePath+fileName)
    writeTextToFile(article.text, filePath, fileName)
    return 1

def writeTextToFile(text, path, fileName):
    try:
        os.makedirs(path, exist_ok=True)
        f = codecs.open(path+fileName, "w", "utf-8")
        f.write(text)
        f.close()
    except:
        print("failed to write: "+fileName)

DATA_DIR = "../../data/raw/data/"

getPapers(all_news, exclusion_set)
