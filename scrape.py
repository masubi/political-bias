import newspaper
from newspaper import Article

real_news = [("http://cnn.com", ["http://cnnespanol.cnn.com"]),
 ("http://reuters.com", [])]

for paper in real_news:
    paper = newspaper.build(paper[0], language ='en', memoize_articles=False)
    for article in paper.articles:
        print(article.url)
    print(paper.size)


#cnn_paper = newspaper.build('http://cnn.com', language='es', memoize_articles=False)

#print("paperSize"+str(cnn_paper.size))

#for article in cnn_paper.articles:
#    print(article.url)

#print("###############")

#for category in cnn_paper.category_urls():
#     print(category)

def downloadArticle(url):
    article = Article(url)
    article.download()
    article.parse()
    print(article.text)



#paper = newspaper.build('https://www.breitbart.com/', memoize_articles=False)
#paper = newspaper.build('https://www.cnn.com/')
#paper = newspaper.build('https://www.infowars.com/')
#paper = newspaper.build('https://www.foxnews.com/', memoize_articles=False)
#print(paper.size())

#for article in paper.articles:
#    print(article.url)

#print("end!")
