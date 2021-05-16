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
    if event == 'typhoon':
        cnt = 0
    for word in myWords[event]:
        print(word + ':', end=' ')
        try:
            if event == 'typhoon' and cnt < 5:
                similar_words[event] += model.most_similar(word)[:3]
                print(similar_words[event][-3:])
                cnt += 1
            else:
                similar_words[event] += model.most_similar(word)[:2]
                print(similar_words[event][-2:])
        except Exception as e:
            print(colored('Not Found', 'red'))

# pre process similar words
unique = {}
for event in similar_words:
    text = ' '.join([word[0] for word in similar_words[event]])
    text_tokens = remove_two_letter_words_from_tokens(remove_non_ascii_chars_from_tokenz(lemmatization_tokenize(remove_stop_words_tokenize(
        text_lowercase(
            remove_numbers(
                remove_punctuation(remove_html_entities(remove_rt(remove_urls(remove_mentions(text)))),
                                   translator))),
        stop_words), lemmatizer)))
    temp_words = list(myWords[event]) + text_tokens
    unique[event] = uniq(temp_words)
    unique[event] = unique[event][:45]
    # similar_words[event] = list(set(text_tokens))[:20]

df = pd.DataFrame.from_dict(unique, orient='index')
df = df.transpose()
df.to_csv(allDatasets + 'similar_words_TF-IDF_Top20.csv', index=False, encoding='utf-8')
print(colored('Done!', 'green'))
