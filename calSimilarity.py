# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1399/10/20        #
# # # # # # # # # # # # # # # # #

import gensim
from gensim.models import Word2Vec
import gensim.downloader
import numpy as np
from preprocessing import *

print(colored('Reading Glove twitter 100 model...', 'blue'))
model = gensim.downloader.load('glove-twitter-100')

allDatasets = 'Datasets/CrisisLexT26/'
event_vector = pd.read_csv(allDatasets + 'TF-IDF_Top10.csv', encoding='utf-8')

out_df = pd.DataFrame(columns=['building_collapse', 'crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon', 'wildfire', 'Label', 'Filename', 'Index', 'Tweet_text'])
print(out_df)

for event in list(event_vector):
    print(colored(event, 'blue') + ':')
    for dataset in sorted(listdir(allDatasets + event)):
        filename = allDatasets + event + '/' + dataset + '/' + listdir(allDatasets + event + '/' + dataset)[2]
        df = pd.read_csv(filename, encoding='utf-8', lineterminator='\n')
        print(filename, len(df))

        for index, row in df.iterrows():
            if row[4] == 'Related and informative' or row[4] == 'Related - but not informative':
                tweet_text = row[1]
                tweet_vector = remove_two_letter_words_from_tokens(remove_non_ascii_chars_from_tokenz(lemmatization_tokenize(remove_stop_words_tokenize(
                    text_lowercase(
                        remove_numbers(
                            remove_punctuation(remove_html_entities(remove_rt(remove_urls(remove_mentions(tweet_text)))),
                                               translator))),
                    stop_words), lemmatizer)))
                # print(tweet_vector)

                out_df_row = {}
                for event2 in list(event_vector):
                    similarity_matrix = np.zeros([len(tweet_vector), len(event_vector[event2])])
                    for i, tweet_word in enumerate(tweet_vector):
                        try:
                            for j, event_word in enumerate(event_vector[event2]):
                                similarity_matrix[i, j] = model.similarity(tweet_word, event_word)
                        except Exception as e:
                            continue

                    sum = 0
                    for i, tweet_word in enumerate(tweet_vector):
                        sum += max(similarity_matrix[i,:])

                    similarity = sum / len(tweet_vector)
                    out_df_row[event2] = similarity

                out_df_row['Label'] = event
                out_df_row['Filename'] = listdir(allDatasets + event + '/' + dataset)[2]
                out_df_row['Index'] = index
                out_df_row['Tweet_text'] = tweet_text
                out_df = out_df.append(out_df_row, ignore_index=True)

out_df.to_csv(allDatasets + 'Similarity_TF-IDF_Top10.csv', index=False, encoding='utf-8')
print(colored('Done!', 'green'))
