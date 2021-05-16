# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# By: Ali Gandomi                                                           #
# Created at: 1399/10/26                                                    #
# Using: https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
from os import listdir
from os.path import isfile, join
from termcolor import colored
import html


def print_df(df):
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(index, 'of', len(df))
        # print(str(index) + ':', colored('id:', 'cyan'), row['id'], colored('user_id:', 'cyan'), row['user_id'], colored('created_at:', 'cyan'), row['created_at'], colored('text:', 'cyan'), row['text'])
        print(str(index) + ':', row['text'])
        if index == 100:
            break
    print('\n\n\n')


def remove_mentions(text):
    # Using: https://stackoverflow.com/a/50830588
    # print(colored('remove_mentions:', 'magenta'), end=' ')
    out = re.sub('@[^\s]+', ' ', text)
    # print(out)
    return out


def remove_urls(text):
    # Using: https://stackoverflow.com/a/11332580
    # print(colored('remove_urls:', 'magenta'), end=' ')
    out = re.sub(r'http://\S+|https://\S+', ' ', text)
    # print(out)
    return out


def remove_rt(text):
    # print(colored('remove_rt:', 'magenta'), end=' ')
    out = re.sub(r'^RT | RT | RT$', ' ', text)
    # print(out)
    return out


def remove_html_entities(text):
    # print(colored('remove_html_entities:', 'magenta'), end=' ')
    out = html.unescape(text)
    # print(out)
    return out


def remove_punctuation(text, translator):
    # print(colored('remove_punctuation:', 'magenta'), end=' ')
    out = text.translate(translator)
    # print(out)
    return out


def remove_numbers(text):
    # print(colored('remove_numbers:', 'magenta'), end=' ')
    out = re.sub(r'\d+', ' ', text)
    # print(out)
    return out


def text_lowercase(text):
    # print(str(index) + ':', colored('id:', 'cyan'), row['id'], colored('user_id:', 'cyan'), row['user_id'], colored('created_at:', 'cyan'), row['created_at'], colored('text:', 'cyan'), row['text'])
    # print(str(index) + ':', df.at[index, 'text'])
    # print(str(index) + ':', df.at[index, 'text'])
    # print(colored('text_lowercase:', 'magenta'), end=' ')
    out = text.lower()
    # print(out)
    return out


def remove_stop_words(text, stop_words):
    # print(colored('remove_stop_words:', 'magenta'), end=' ')
    word_tokens = word_tokenize(text)
    out = ' '.join([word for word in word_tokens if word not in stop_words])
    # print(out)
    return out


def remove_stop_words_tokenize(text, stop_words):
    # print(colored('remove_stop_words:', 'magenta'), end=' ')
    word_tokens = word_tokenize(text)
    out = [word for word in word_tokens if word not in stop_words]
    # print(out)
    return out


def lemmatization(text, lemmatizer):
    # print(colored('lemmatization:', 'magenta'), end=' ')
    word_tokens = word_tokenize(text)
    out = ' '.join([lemmatizer.lemmatize(word, pos='v') for word in word_tokens])
    # print(out)
    return out


def lemmatization_tokenize(tokens, lemmatizer):
    # print(colored('lemmatization:', 'magenta'), end=' ')
    out = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    # print(out)
    return out


def remove_non_ascii_chars(text):
    """
    return text after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    return ''.join([w if ord(w) < 128 else ' ' for w in text])


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def remove_non_ascii_chars_from_tokenz(tokens):
    """
    return not empty tokens after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    for idx, word in enumerate(tokens):
        tokens[idx] = remove_non_ascii_chars(word)
    out = [word for word in tokens if word != '']
    return out


def remove_two_letter_words(text):
    # print(colored('remove_two_letter_words:', 'magenta'), end=' ')
    word_tokens = word_tokenize(text)
    out = ' '.join([word for word in word_tokens if len(word) > 2])
    # print(out)
    return out


def remove_two_letter_words_from_tokens(tokens):
    # print(colored('remove_two_letter_words_from_tokens:', 'magenta'), end=' ')
    out = [word for word in tokens if len(word) > 2]
    # print(out)
    return out


translator = str.maketrans('', '', string.punctuation)
# translator = str.maketrans(string.punctuation, '                                ')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


if __name__ == "__main__":
    Datasets = [
        'FA Cup Final',
        'Super Tuesday for US Elections',
        'US Elections',
    ]
    Dataset_folder = 'Datasets'
    Tweets_folder = 'Tweets'
    Preprocessed_folder = 'Preprocessed'

    for Dataset in Datasets:
        print(colored('Dataset:', 'blue'), Dataset)
        Tweets_path = Dataset_folder + '/' + Dataset + '/' + Tweets_folder + '/'
        Preprocessed_path = Dataset_folder + '/' + Dataset + '/' + Preprocessed_folder + '/'

        Tweets_files = [f for f in listdir(Tweets_path) if isfile(join(Tweets_path, f))]
        print(colored('# files:', 'cyan'), len(Tweets_files))
        df = pd.DataFrame([])
        for index, Tweets_file in enumerate(Tweets_files):
            # print(str(index) + ':', '-- '+colored('filename:', 'cyan'), Tweets_file, end=' ')
            # عبارت lineterminator برای اینه که r\ به عنوان پایان خط در نظر گرفته نشود و فقط n\ برای پایان خط در نظر گرفته شود
            current_df = pd.read_csv(Tweets_path + Tweets_file, encoding='utf-8', lineterminator='\n')
            df = pd.concat([df, current_df], ignore_index=True)
            # print('--', colored('# Tweets:', 'cyan'), len(current_df), '-- '+colored('# all:', 'cyan'), len(df))

        for index, row in df.iterrows():
            # print(index, 'of', str(len(df)) + ':', row['text'])
            if index % 10000 == 0:
                print(index, 'of', str(len(df)))
            df.at[index, 'text'] = lemmatization(remove_stop_words(
                text_lowercase(
                    remove_numbers(
                        remove_punctuation(remove_rt(remove_urls(remove_mentions(row['text']))), translator))),
                stop_words), lemmatizer)
            # print('\n')

        df.to_csv(Preprocessed_path + Dataset + '.csv', index=False, encoding='utf-8')

    print(colored('Done!', 'green'))
