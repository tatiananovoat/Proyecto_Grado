from django.shortcuts import render, render_to_response, HttpResponse
from core.functions import ObtieneTweets, LimpiarTextoTweets, PlotFrecuencia, PlotWordCloud, PlotClustering, PlotAnalisisSentimientos, PlotMaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import Counter
import pandas as pd
import numpy as np
import re

def Buscar_Twitter(request, palabras=None):

    Tweets = ObtieneTweets(palabras, 1000)
    Tweets = LimpiarTextoTweets(Tweets, palabras)

    #TweetsMostrar = Tweets[['date', 'text', 'tokens', 'removed', 'grams', 'AnalisisSentimientos']]
    TweetsMostrar = Tweets[['date', 'text', 'AnalisisSentimientos']]
    TweetsMostrar = TweetsMostrar[:200]

    html_table = TweetsMostrar.to_html(
        index=False, classes='footable table table-stripped toggle-arrow-tiny')
    html_table = re.sub(
        r'<table([^>]*)>',
        r'<table\1 data-page-size="10" data-filter=#filter>',
        html_table)

    html_table = html_table.replace(
        "</table>", "<tfoot><tr><td colspan=\"12\"><ul class=\"pagination float-right\"></ul></td></tr></tfoot></table>")
    responseImage = PlotFrecuencia(Tweets)
    responseWordCloud = PlotWordCloud(Tweets)
    responseClustering = PlotClustering(Tweets, palabras)
    responseSentimientos = PlotAnalisisSentimientos(Tweets)
    responseMap = PlotMaps(Tweets)

    Dict = {'html_table': html_table, 'responseImage': responseImage, 
    'responseWordCloud': responseWordCloud, 'responseSentimientos' : responseSentimientos, 'responseMap' : responseMap}

    for t in range(0,len(responseClustering)):
        Dict['responseClustering'+str(t)]=responseClustering[t]
        print('responseClustering'+str(t))

    return render(request, 'core/Buscar_Twitter.html', Dict)

def Inicio_Buscar(request):
    return render(request, 'core/Analisis_Twitter.html')

#def Submit_Buscar(self):
    #Valor_Busqueda= request.POST.get('valor')

