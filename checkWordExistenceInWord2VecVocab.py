# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1399/10/20        #
# # # # # # # # # # # # # # # # #

from gensim.models import Word2Vec
import gensim.downloader
from termcolor import colored
import pandas as pd

print(colored('Reading Glove twitter 100 model...', 'blue'))
model = gensim.downloader.load('glove-twitter-100')

allDatasets = 'Datasets/CrisisLexT26/'
myWords = pd.read_csv(allDatasets + 'similar_words_TF-IDF_Top20.csv', encoding='utf-8')

for event in list(myWords):
    print(colored(event, 'blue') + ':')
    part2 = False
    for word in myWords[event]:

        print(colored(word, 'blue') + ':', end=' ')

        try:
            print(model.most_similar(word))
        except Exception as e:
            print(colored('Not Found', 'red'))

print(colored('Done!', 'green'))
