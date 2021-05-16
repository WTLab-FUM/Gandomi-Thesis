# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1399/12/24        #
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

warnings.simplefilter("ignore")

print('Reading Glove twitter 100 - classifier models & events vectors...')
classifier_model = pickle.load(open('XGBoost.model', 'rb'))
event_vector = pd.read_csv('TF-IDF_Top20.csv', encoding='utf-8')
model = gensim.downloader.load('glove-twitter-100')
print('Ready...')

decoding = ['building_collapse', 'crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon',
            'wildfire']

cnt = 1
while True:
    filenames = glob.glob('data_*.csv')
    try:
        if len(filenames) != 0:
            for filename in filenames:
                print(str(cnt)+'.')
                print('File:', filename)

                f = open(filename, encoding="utf8")
                tweet_text = f.read()
                f.close()
                tweet_vector = remove_two_letter_words_from_tokens(remove_non_ascii_chars_from_tokenz(lemmatization_tokenize(remove_stop_words_tokenize(
                    text_lowercase(
                        remove_numbers(
                            remove_punctuation(remove_html_entities(remove_rt(remove_urls(remove_mentions(tweet_text)))),
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
                r = redis.Redis()
                r.set(filename[5:-4], str(result))
                # output_filename = 'output_'+filename[5:-4]+'.txt'
                # file = open(output_filename, 'w')
                # file.write(str(result))
                # file.close()
                print('Result:', decoding[result])
                # print('Output filename:', output_filename, end='\n\n')

                os.remove(filename)
                cnt += 1
        else:
            time.sleep(0.1)

    except Exception as e:
        # print(e)
        print(traceback.format_exc())
        time.sleep(0.1)
