# # # # # # # # # # # # # # # # #
# By: Ali Gandomi               #
# Created at: 1400/01/31        #
# # # # # # # # # # # # # # # # #

import gensim.downloader
import numpy as np
import pandas as pd
import pickle
import requests
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_fscore_support as score

dataset = 'Datasets/CrisisLexT26/'

event_vector = pd.read_csv('TF-IDF_Top20.csv', encoding='utf-8')
print('Ready...')

decoding = ['building_collapse', 'crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon',
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


test_dataset = pd.read_csv('test.csv', encoding='utf-8')
X_test = test_dataset.iloc[:, 13]
y_test = test_dataset.iloc[:, 10]
y_test = y_test.values.ravel()

predicted = np.empty(shape=len(y_test))
url = 'http://127.0.0.1/Healthfog/myThesis.php'
for index, tweet in enumerate(X_test):
    data = {'data': tweet}
    x = requests.post(url, data=data)
    predicted[index] = x.text
    print(x.text)

evaluation(y_test, predicted)
