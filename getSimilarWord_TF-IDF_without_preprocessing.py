# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/01/26        #
# # # # # # # # # # # # # # # # #

import gensim.downloader
from preprocessing import *


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


print(colored('Reading Glove twitter 100 model...', 'blue'))
model = gensim.downloader.load('glove-twitter-100')
allDatasets = 'Datasets/CrisisLexT26/'
myWords = pd.read_csv(allDatasets + 'TF-IDF_Top20.csv', encoding='utf-8')

similar_words = {}
for event in list(myWords):
    print(colored(event, 'blue') + ':')
    similar_words[event] = []
    for word in myWords[event]:
        print(word + ':', end=' ')
        try:
            similar_words[event] += [model.most_similar(word)[:1][0][0]]
            print(similar_words[event][-1:])
        except Exception as e:
            print(colored('Not Found', 'red'))

all_word = {}
for event in similar_words:
    all_word[event] = list(myWords[event]) + similar_words[event]

df = pd.DataFrame.from_dict(all_word, orient='index')
df = df.transpose()
df.to_csv(allDatasets + 'similar_words_TF-IDF_Top20_without_preprocessing.csv', index=False, encoding='utf-8')
print(colored('Done!', 'green'))
