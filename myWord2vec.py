# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1399/10/26        #
# # # # # # # # # # # # # # # # #

import gensim
import bz2
import json
from preprocessing import *


class MySentences(object):
    def __init__(self, dirname):
        self.allDatasets = dirname
        self.event_vector = pd.read_csv(self.allDatasets + 'TF-IDF_Top20.csv', encoding='utf-8')
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        for event in list(self.event_vector):
            print(colored('epoch ' + str(self.epoch) + ':', 'blue'), colored(event, 'blue') + ':')
            for dataset in sorted(listdir(self.allDatasets + event)):
                filename = self.allDatasets + event + '/' + dataset + '/' + listdir(self.allDatasets + event + '/' + dataset)[2]
                df = pd.read_csv(filename, encoding='utf-8', lineterminator='\n')
                print(colored('epoch ' + str(self.epoch) + ':', 'blue'), filename, len(df))

                for index, row in df.iterrows():
                    if row[4] == 'Related and informative' or row[4] == 'Related - but not informative':
                        tweet_text = row[1]
                        tweet_vector = remove_non_ascii_chars_from_tokenz(
                            lemmatization_tokenize(remove_stop_words_tokenize(
                                text_lowercase(
                                    remove_numbers(
                                        remove_punctuation(remove_rt(remove_urls(remove_mentions(tweet_text))),
                                                           translator))),
                                stop_words), lemmatizer))
                        # print(tweet_vector)
                        yield tweet_vector


sentences = MySentences('Datasets/CrisisLexT26/')
model = gensim.models.Word2Vec(sentences, sg=1, iter=5, window=6, seed=42, workers=7)
model.save('Datasets/CrisisLexT26/Models/Word2Vec/word2vec_skip-gram.wordVectors')
print(colored('Done!', 'green'))
