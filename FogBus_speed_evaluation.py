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
import time

decoding = ['building_collapse', 'crash', 'earthquake', 'explosion', 'flood', 'haze', 'meteor', 'shoot', 'typhoon',
            'wildfire']


test_dataset = pd.read_csv('test.csv', encoding='utf-8')
X_test = test_dataset.iloc[:, 13]
y_test = test_dataset.iloc[:, 10]
y_test = y_test.values.ravel()

predicted = np.empty(shape=len(y_test))
url = 'http://127.0.0.1/Healthfog/myThesis.php'
elapsed_times = np.zeros(1000)
for index, tweet in enumerate(X_test):
    data = {'data': tweet}
    start_time = time.time()
    x = requests.post(url, data=data)
    elapsed_times[index] = (time.time() - start_time) * 1000
    print(str(index) + '.', elapsed_times[index], 'milliseconds ---', np.mean(elapsed_times[1:index+1]), 'average')
    predicted[index] = x.text
    print(decoding[int(x.text)])

    if index == 999:
        break
