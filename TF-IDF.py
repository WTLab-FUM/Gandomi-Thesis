# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/01/26        #
# # # # # # # # # # # # # # # # #

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *

def scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores

allDatasets = 'Datasets/CrisisLexT26/'

event_vector = {
    'building_collapse': [],
    'crash': [],
    'earthquake': [],
    'explosion': [],
    'flood': [],
    'haze': [],
    'meteor': [],
    'shoot': [],
    'typhoon': [],
    'wildfire': [],
}

for event in event_vector:
    print(colored(event, 'blue') + ':')
    corpus = []
    for dataset in sorted(listdir(allDatasets + event)):
        filename = allDatasets + event + '/' + dataset + '/' + listdir(allDatasets + event + '/' + dataset)[2]
        df = pd.read_csv(filename, encoding='utf-8', lineterminator='\n')
        print(filename, len(df))

        for index, row in df.iterrows():
            tweet_text = row[1]
            tweet_preprocessed = remove_two_letter_words(remove_non_ascii_chars(lemmatization(remove_stop_words(
                text_lowercase(
                    remove_numbers(
                        remove_punctuation(remove_html_entities(remove_rt(remove_urls(remove_mentions(tweet_text)))),
                                           translator))),
                stop_words), lemmatizer)))
            corpus += [tweet_preprocessed]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    # feature_names = vectorizer.get_feature_names()
    # dense = vectors.todense()
    # denselist = dense.tolist()
    # df = pd.DataFrame(denselist, columns=feature_names)
    # df.to_csv(allDatasets + event + '/' + 'TF-IDF.csv', index=False, encoding='utf-8')
    # event_vector[event] = df
    event_vector[event] = scores(vectorizer, vectors)

df = pd.DataFrame.from_dict(event_vector, orient='index')
df = df.transpose()
df.to_csv(allDatasets + 'TF-IDF.csv', index=False, encoding='utf-8')
print(colored('Done!', 'green'))
