import newspaper
from newspaper import Article
from newspaper import news_pool


debugMode = False
memoize = True

def log(msg):
    print(msg)

def debug(msg):
    if(debugMode):
        print(msg)

# tuple (site, set(exclusion list))
real_news = [#"http://cnn.com",
# "http://reuters.com",
# "https://www.bbc.com",
# "https://www.npr.org",
# "https://www.nytimes.com",
# "https://www.wsj.com",
# "https://www.washingtonpost.com",
# "https://www.economist.com",
# "https://www.ap.org",
# "https://www.theatlantic.com",
# "https://gizmodo.com/",
# "https://www.wired.com/",
# "https://www.nature.com/",
 "https://www.vox.com/",

  # Returns 0 Articles
  #"https://www.forbes.com",
  #"https://www.bloomberg.com",
 ]

fake_news = ["https://www.breitbart.com",
"https://www.infowars.com",
"https://www.foxnews.com"
]

exclusion_set = set(["cnnespanol.cnn.com","arabic.cnn.com" "br.reuters.com","fr.reuters.com","es.reuters.com", "it.reuters.com","cn.reuters.com","reuters.zendesk.com","ru.reuters.com","ara.reuters.com","de.reuters.com","ar.reuters.com","mx.reuters.com","jp.reuters.com",
"cn.nytimes.com"])

def processNewsPaper(paper):
    targetPaper = newspaper.build(paper, language ='en', memoize_articles=memoize)

    articlesForPaperIncluded = 0
    for article in targetPaper.articles:
        # check if in exclusion list
        checkDomainName = article.url.split("/")[2]
        debug("check exclusion_set: "+ checkDomainName)
        if(checkDomainName in exclusionSet):
            debug("excluding: "+article.url)
        else:
            log(article.url)
            articlesForPaperIncluded=articlesForPaperIncluded+1
            totalArticlesIncluded=totalArticlesIncluded+1

    log("---------------------------")
    log("paper: " + paper)
    log("articlesForPaperIncluded: " + str(articlesForPaperIncluded))
    log("totalPaper articles: " + str(targetPaper.size()))
    log("---------------------------")

    return articlesForPaperIncluded

def getPapers(newspapers, exclusionSet):
    totalArticlesIncluded = 0
    for paper in newspapers:
        targetPaper = newspaper.build(paper, language ='en', memoize_articles=memoize)

        articlesForPaperIncluded = 0
        for article in targetPaper.articles:
            # check if in exclusion list
            checkDomainName = article.url.split("/")[2]
            debug("check exclusion_set: "+ checkDomainName)
            if(checkDomainName in exclusionSet):
                debug("excluding: "+article.url)
            else:
                log(article.url)
                articlesForPaperIncluded=articlesForPaperIncluded+1
                totalArticlesIncluded=totalArticlesIncluded+1

        log("---------------------------")
        log("paper: " + paper)
        log("articlesForPaperIncluded: " + str(articlesForPaperIncluded))
        log("totalPaper articles: " + str(targetPaper.size()))
        log("---------------------------")

    log("totalArticlesIncluded: "+str(totalArticlesIncluded))

def downloadArticle(url):
    article = Article(url)
    article.download()
    article.parse()
    print(article.text) # TODO:  download some place

def writeTextToFile(text, fileName):
    f = open(fileName, "a")
    f.write(text)
    f.close()

getPapers(real_news, exclusion_set)
#getPapers(fake_news, exclusion_set)

# examples
#cnn_paper = newspaper.build('http://cnn.com', language='es', memoize_articles=False)

#print("paperSize"+str(cnn_paper.size))

#for article in cnn_paper.articles:
#    print(article.url)

#print("###############")

#for category in cnn_paper.category_urls():
#     print(category)
