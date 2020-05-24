from twython import Twython
from django.shortcuts import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage
from mpl_toolkits.basemap import Basemap
from datetime import date
import classifier 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import urllib, base64
import pandas as pd
import numpy as np
import unidecode
import random
import nltk
import json
import io
import os


def ObtieneTweets(Busqueda, Cantidad):
    
    Consultar = 0
    credentials = {}  
    credentials['CONSUMER_KEY'] = 'oRSMrRLW5LTnmxCYNI0TBxKAt'  
    credentials['CONSUMER_SECRET'] = 'wSqv5dzvEPCXYvOh9Te7sptXonMxaHWJ4cZsHQySNLrO1kNUe5'
    credentials['ACCESS_TOKEN'] = '1034819063824965633-zVmUkaSSDQJRmizgMoMomYMcZsyfb3'  
    credentials['ACCESS_SECRET'] = 'GLS13zAeK1y3KcDbmcD1VddvfnxOrI3TgSv4nk0GW5O6C'
    python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'],credentials['ACCESS_TOKEN'],credentials['ACCESS_SECRET'])
    
    t = str(date.today())
    
    dir = os.path.dirname(__file__)
    filename = dir + '\\data\\Tweets_'+ Busqueda.lower()

    if Consultar == 1:
        search= python_tweets.search(q=Busqueda,count=Cantidad,lang='es')['statuses']
        os.makedirs(filename,exist_ok=True)

        for result in search:   
            with open(filename +'\\Tweets_'+ t +'.txt', 'a') as outfile:
                json.dump(result, outfile, sort_keys = True, indent = 4)

    with open(filename +'\\Tweets_'+ t +'.txt') as f:
        content = f.read()
        content = content.replace('}{','},{')
        search = json.loads('['+content+']')

    dict_ = {'user': [], 'date': [], 'text': [], 'text_clean': [], 'loc':[], 'favorite_count': [],
             'Busqueda':[], 'ValorAnalisisSentimientos':[], 'AnalisisSentimientos':[], 'latitud':[], 
             'longitud':[]} 
    
    CantTweets = 0

    dfcities = pd.read_csv(dir + '\\resource\\worldcities.csv')

    for status in search:  
        #print(status)
        dict_['user'].append(status['user']['screen_name'])
        dict_['loc'].append(status['user']['location'])
        dict_['date'].append(status['created_at'])
        dict_['text'].append(unidecode.unidecode(status['text']))
        dict_['text_clean'].append(status['text'])
        dict_['favorite_count'].append(status['favorite_count'])
        #dict_['hashtags'].append([hashtag['text'] for hashtag in status['entities']['hashtags']])
        dict_['Busqueda'].append(Busqueda)
        #dict_['Fecha'].append(Fecha)

        valrandom = random.randrange(1, 15492)
        valrandom

        dict_['latitud'].append(dfcities.loc[valrandom].lat)
        dict_['longitud'].append(dfcities.loc[valrandom].lng)

        CantTweets= CantTweets+1

        try:
            analysis = TextBlob(status['text']).translate(to='en')
        except:
            analysis = TextBlob(status['text'])
        
        dict_['ValorAnalisisSentimientos'].append(analysis.sentiment.polarity)
        if analysis.sentiment.polarity > 0: 
            dict_['AnalisisSentimientos'].append('Positivo')
        elif analysis.sentiment.polarity == 0: 
            dict_['AnalisisSentimientos'].append('Neutral')
        elif analysis.sentiment.polarity < 0: 
            dict_['AnalisisSentimientos'].append('Negativo')
        
        print(str(CantTweets) + '-> ' + str(analysis.sentiment.polarity))
        
    df = pd.DataFrame(dict_) 
    return df

def LimpiarTextoTweets(df, Busqueda):

    #spark = SparkSession.builder.master('spark://192.168.55.3:7077').appName('LimpiaDatos').getOrCreate()
    spark = SparkSession.builder.appName('LimpiaDatos').getOrCreate()

    sdf = spark.createDataFrame(df)
    
    stopword_unidecode = [unidecode.unidecode(word) for word in stopwords.words('spanish')]
    numeros = ['1','2','3','4','5','6','7','8','9','0']

    stopwordList = list(numeros + stopword_unidecode + stopwords.words('spanish')+['rt','https','co','http', 't', 'q', 'l', 'c']+Busqueda.lower().split())
    
    #dataTweet = spark.createDataFrame([(0, unidecode.unidecode(Texto))],['id','sentence'])

    tokenizer = RegexTokenizer(inputCol='text',outputCol='tokens', pattern='\W+')
    tokenized = tokenizer.transform(sdf)
    #tokenized.show(truncate=False)
    
    remover = StopWordsRemover(inputCol='tokens',outputCol='removed', stopWords=stopwordList)
    removered = remover.transform(tokenized)
    #removered.show(truncate=False)
    
    ngram = NGram(n=2, inputCol='removed',outputCol='grams')
    ngramd = ngram.transform(removered)

    Tweets_Limpios = ngramd.toPandas()

    spark.stop()
    
    return Tweets_Limpios   

def PlotAnalisisSentimientos(Tweets):

    counts = Counter(Tweets['AnalisisSentimientos'])
    colors = ['gold', 'lightcoral', 'yellowgreen']

    f = pyplot.figure(figsize=(8,5))
    
    patches, texts, autotexts  = pyplot.pie([float(v) for v in counts.values()], colors=colors, shadow=True, autopct=autopct_fun([float(v) for v in counts.values()]))
    pyplot.legend(patches, [k for k in counts.keys()], loc="best")
    pyplot.axis('off')

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(f)
    canvas.print_png(buf)
    f.savefig(buf, format='png')
    buf.seek(0)

    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    f.clear()
    pyplot.close()

    return uri

def PlotFrecuencia(Tweets):

    tokens = [t for t in np.concatenate(Tweets['removed'])]
    counter = Counter(tokens)

    aDict = {}
    for letter, count in counter.most_common(30):
        aDict[letter] = count

    counterFinal = Counter(aDict)
    words = counterFinal.keys()
    counts = counterFinal.values()
    indexes = np.arange(len(words))

    error = np.random.rand(len(words))

    f, ax = pyplot.subplots(figsize=(14,6))
    ax.barh(indexes, counts, xerr=error, align='center')
    ax.set_yticks(indexes)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title('Frecuencia de Palabras')

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(f)
    canvas.print_png(buf)
    f.savefig(buf, format='png')
    buf.seek(0)

    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    f.clear()
    pyplot.close()

    return uri

def PlotWordCloud(Tweets):

    Tweets_string = " ".join([t for t in np.concatenate(Tweets['removed'])])
    wordcloud = WordCloud().generate(Tweets_string)
    f = pyplot.figure(figsize=(8,4))
    pyplot.imshow(wordcloud)
    pyplot.axis("off")

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(f)
    canvas.print_png(buf)
    f.savefig(buf, format='png')
    buf.seek(0)

    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    f.clear()
    pyplot.close()

    return uri

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        stemmer = nltk.stem.SnowballStemmer('spanish')
        analyzer = super().build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

def post_cluster(vectorized,verbose=0):
    distortions = []
    K = range(1,50)
    for k in K:
        km = KMeans(n_clusters=k, n_init=1, verbose=verbose, random_state=3)
        km.fit(vectorized)
        distortions.append(km.inertia_)
    
    num_clusters = ValNumClusters(distortions)
    km = KMeans(n_clusters=num_clusters, n_init=1, verbose=verbose, random_state=3)
    km.fit(vectorized)

    return km

def groupwordcloud(data,category,stopwordList):
    a=np.array(data)
    urls=[]
    for  i in set(category):
        f, ax = pyplot.subplots(figsize=(14,6))
        wordcloud = WordCloud(width=1200,height=1200,stopwords=stopwordList).generate('\n'.join(a[category==i]))
        pyplot.imshow(wordcloud, interpolation='bilinear', aspect='auto')
        pyplot.axis("off")
        
        buf = io.BytesIO()
        canvas = FigureCanvasAgg(f)
        canvas.print_png(buf)
        f.savefig(buf, format='png')
        buf.seek(0)

        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        urls.append(uri)

        f.clear()
        pyplot.close()

    return urls

def PlotClustering(Tweets, Busqueda):
    
    stopword_unidecode = [unidecode.unidecode(word) for word in stopwords.words('spanish')]
    stopwordList = list(stopword_unidecode + stopwords.words('spanish')+['rt','https','co','http', 't', 'q', 'l', 'c']+Busqueda.lower().split())
    
    vectorizer = StemmedTfidfVectorizer(min_df=2, max_df=0.5,
                                    stop_words=stopwordList, decode_error='ignore',strip_accents ='unicode'
                                    )
    
    vectorized = vectorizer.fit_transform(Tweets['text'].tolist())

    km=post_cluster(vectorized,verbose=1)
    urls = groupwordcloud(Tweets['text'].tolist(),km.labels_,stopwordList)
    return urls

def autopct_fun(abs_values):
    gen = iter(abs_values)
    return lambda pct: f"{pct:.1f}% ({next(gen)})" 

def ValNumClusters(distortions):
    maxValue = 0
    maxIterarion = 0

    X = [[i] for i in distortions]
    Z = linkage(X, metric='euclidean', method='ward')
    
    last = Z[-10:, 2]
    num_clustres = np.arange(1, len(last) + 1)
    gap = np.diff(last, n=2)  
    t = num_clustres[:-2] + 1, gap[::-1]
    
    for vuelta in range(0,len(t[1])):
        if (t[1][vuelta] > maxIterarion):
            maxValue = t[0][vuelta]
            maxIterarion = t[1][vuelta]
    
    return maxValue

def PlotMaps(Tweets):

    latitud = Tweets.latitud.tolist()
    longitud = Tweets.longitud.tolist()

    f = pyplot.figure(figsize=(8,6))

    map = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
    map.drawmapboundary(fill_color='aqua')
    map.fillcontinents(color='white',lake_color='aqua')
    map.drawcountries()

    xpt,ypt = map(longitud,latitud)
    map.plot(xpt,ypt,'or')

    #pyplot.imshow(map)
    pyplot.axis("off")

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(f)
    canvas.print_png(buf)
    f.savefig(buf, format='png')
    buf.seek(0)

    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    f.clear()
    pyplot.close()

    return uri


    
