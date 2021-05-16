# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/01/31        #
# # # # # # # # # # # # # # # # #

import os
import time
import pandas as pd
import gensim.downloader
import numpy as np
from preprocessing import *
import pickle
import sys
import warnings
import glob
import traceback
import redis
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support as score

warnings.simplefilter("ignore")

dataset = 'Datasets/CrisisLexT26/'

print('Reading Glove tweeter 100 - classifier models & events vectors...')
model = gensim.downloader.load('glove-twitter-100')
classifier_model = pickle.load(open('Datasets/CrisisLexT26/Models/XGBoost/XGBoost.model', 'rb'))
event_vector = pd.read_csv(dataset + 'TF-IDF_Top20.csv', encoding='utf-8')
print('Ready...')

decoding = ['crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon',
            'wildfire']

def evaluation(y_test, predicted):
    accuracy = (np.sum(np.array(y_test) == np.array(predicted)) / len(y_test)) * 100
    print('Accuracy:', accuracy, '%')
    # precision recall fscore
    precision, recall, fscore, support = score(y_test, predicted)
    x = PrettyTable()
    # x.field_names = [model_name, "Precision", "Recall", "FScore", "Support"]
    x.field_names = ["Label", "Precision", "Recall", "FScore"]
    for index, key in enumerate(decoding):
        # x.add_row([key, str(int(precision[index] * 10000) / 100) + '%', str(int(recall[index] * 10000) / 100) + '%',
        #            str(int(fscore[index] * 10000) / 100) + '%', support[index]])
        x.add_row([key, str(int(precision[index] * 10000) / 100) + '%', str(int(recall[index] * 10000) / 100) + '%',
                   str(int(fscore[index] * 10000) / 100) + '%'])
    print(x)
    x.add_row(
        ['average', str(int(np.mean(precision) * 10000) / 100) + '%', str(int(np.mean(recall) * 10000) / 100) + '%',
         str(int(np.mean(fscore) * 10000) / 100) + '%'])
    print("\n".join(x.get_string().splitlines()[-2:]))
    return accuracy

test_dataset = pd.read_csv(dataset + 'X_test.csv', encoding='utf-8')
X_test = test_dataset.iloc[:, 12]
y_test = test_dataset.iloc[:, 9]
y_test = y_test.values.ravel()

predicted = np.empty(shape=len(y_test))
for index, tweet in enumerate(X_test):
    print(index, 'of', len(X_test), ':', tweet)
    tweet_vector = remove_two_letter_words_from_tokens(
        remove_non_ascii_chars_from_tokenz(lemmatization_tokenize(remove_stop_words_tokenize(
            text_lowercase(
                remove_numbers(
                    remove_punctuation(remove_rt(remove_urls(remove_mentions(tweet))),
                                       translator))),
            stop_words), lemmatizer)))
    out_df_row = {}
    for event2 in event_vector:
        similarity_matrix = np.zeros([len(tweet_vector), len(event_vector[event2])])
        for i, tweet_word in enumerate(tweet_vector):
            try:
                for j, event_word in enumerate(event_vector[event2]):
                    similarity_matrix[i, j] = model.similarity(tweet_word, event_word)
            except Exception as e:
                continue

        sum = 0
        for i, tweet_word in enumerate(tweet_vector):
            sum += max(similarity_matrix[i, :])

        similarity = sum / len(tweet_vector)
        out_df_row[event2] = [similarity]

    df = pd.DataFrame(out_df_row)
    # X = [df.astype(np.float64)]

    result = classifier_model.predict(df)[0].astype(int)
    predicted[index] = result
    # r = redis.Redis()
    # r.set(filename[5:-4], str(result))
    # output_filename = 'output_'+filename[5:-4]+'.txt'
    # file = open(output_filename, 'w')
    # file.write(str(result))
    # file.close()
    print(decoding[result])
    # print('Output filename:', output_filename, end='\n\n')

evaluation(y_test, predicted)

print(colored('Done!', 'green'))
